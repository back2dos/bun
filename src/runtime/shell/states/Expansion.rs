//! Word expansion: tilde, variable, command substitution, brace, and glob.
//!
//! If a word contains command substitution or glob expansion syntax then it
//! needs to be evaluated at runtime — this state node walks the atom and
//! produces zero or more output strings.

use crate::shell::ast;
use crate::shell::interpreter::{
    EnvMap, EventLoopHandle, Interpreter, Node, NodeId, ShellExecEnv, ShellExecEnvKind, StateKind,
    log,
};
use crate::shell::io::{IO, OutKind};
use crate::shell::states::assigns::{AssignCtx, AssignsState};
use crate::shell::states::base::Base;
use crate::shell::states::script::Script;
use crate::shell::yield_::Yield;
use crate::shell::{EnvStr, ExitCode, ShellErr};

pub(crate) struct Expansion {
    pub(crate) base: Base,
    pub node: bun_ptr::BackRef<ast::Atom>,
    pub(crate) state: ExpansionState,
    /// Index of the next sub-atom to expand. For `Atom::Simple` this is 0/1;
    /// for `Atom::Compound` it walks `c.atoms`.
    pub(crate) word_idx: u32,
    /// Output sink the parent provided. The parent is reachable via
    /// `base.parent`, so the sink is just a buffer the parent reads back on
    /// `child_done`.
    pub out: ExpansionOut,
    /// Working buffer for the *current* word being assembled. When a word
    /// boundary is hit (IFS split / glob result), it is flushed into `out`
    /// via `push_current_out`.
    pub(crate) current_out: Vec<u8>,
    /// Byte offsets in `current_out` written by literal metacharacter atoms
    /// (`Asterisk`/`DoubleAsterisk`/`BraceBegin`/`Comma`/`BraceEnd`). Only
    /// these positions may act as pattern syntax in `do_brace_expand` or
    /// `transition_to_glob_state`; metacharacter bytes from any other source
    /// (JS interpolation, quoted text, `$var`, command substitution) are data
    /// and must not change the expansion structure or broaden the glob.
    pub(crate) meta_offsets: Vec<u32>,
    /// The `Script` of a `$(...)`, or in `ExpansionState::ParamWord` the `Expansion` of a word.
    pub(crate) child_script: Option<NodeId>,
    /// Whether the in-flight command substitution was `"$(...)"` (no IFS
    /// splitting on its result). Only meaningful while `state == CmdSubst`.
    pub(crate) cmd_subst_quoted: bool,
    /// The atom is an assignment value (`A=v`, `export A=v`), so
    /// command-substitution output is not field split (POSIX 2.9.1).
    pub(crate) assign_ctx: bool,
    /// Set when a `""`/`''` literal
    /// was seen so an *empty* expansion is still pushed as an argv word.
    /// Without this, `$unset` and `""` are indistinguishable in
    /// [`ExpansionOut`] (both → `buf=[], bounds=[]`) and Cmd would push an
    /// empty arg for unset vars — diverging from POSIX field-splitting.
    pub(crate) has_quoted_empty: bool,
    /// Exit code of a sole-command-substitution arg — propagated to `Cmd`
    /// so `$(false)` as argv0 fails.
    pub(crate) out_exit_code: ExitCode,
}

#[derive(Default, strum::IntoStaticStr)]
pub enum ExpansionState {
    #[default]
    Idle,
    Walking,
    CmdSubst,
    /// The word of the `${NAME<op>word}` at `word_idx` holds a `$(...)` and expands in a child `Expansion`.
    ParamWord,
    Glob,
    BraceExpand,
    Done,
    /// The parent inspects this on
    /// `child_done(_, 1)` to print the message.
    Err(Box<ShellErr>),
}

/// What [`Expansion::expand_simple_no_io`] leaves to its caller.
enum Step {
    /// Nothing: the atom is expanded.
    Next,
    /// A `$(...)` to run.
    CmdSubst,
    /// A `${NAME<op>word}`.
    Param,
    /// Unquoted whitespace in the word of a `${NAME<op>word}`.
    ParamSpace,
}

/// What a `${NAME<op>word}` reads besides the shell environment.
struct ParamCtx<'a> {
    event_loop: EventLoopHandle,
    command_ctx: *mut bun_options_types::context::ContextData,
    vm_args_utf8: &'a mut Vec<bun_core::Utf8Bytes<'static>>,
    /// In the value of a prefix assignment (`A=1 B=${A:-2} cmd`): the assignments before it.
    earlier_assigns: &'a [ast::Assign],
}

const _: () = assert!(size_of::<Expansion>() == 168);

#[derive(Default)]
pub(crate) struct ExpansionOut {
    pub(crate) buf: Vec<u8>,
    /// Word boundaries within `buf` (for IFS splitting / glob results).
    pub(crate) bounds: Vec<u32>,
    /// Set when the atom is a sole `$(…)`
    /// that exited non-zero, so [`Cmd::child_done`] can propagate it as the
    /// command's exit code when that substitution was argv0 and argv is
    /// otherwise empty.
    pub(crate) out_exit_code: ExitCode,
    /// When `buf`/`bounds` are both
    /// empty, this distinguishes `""` (push one empty arg) from `$unset`
    /// (push no arg). See [`Expansion::has_quoted_empty`].
    pub(crate) has_quoted_empty: bool,
}

impl Expansion {
    pub(crate) fn init(
        interp: &Interpreter,
        shell: *mut ShellExecEnv,
        node: *const ast::Atom,
        parent: NodeId,
        assign_ctx: bool,
    ) -> NodeId {
        interp.alloc_node(Node::Expansion(Expansion {
            base: Base::new(parent, shell),
            // SAFETY: `node` is non-null and points into the AST arena
            // (`ShellArgs::__arena`), which the interpreter holds for its
            // entire lifetime — strictly outliving every state node (the
            // BackRef invariant). Callers pass `&raw const` only to escape
            // borrowck across the `&Interpreter` reborrow.
            node: unsafe { bun_ptr::BackRef::from_raw(node as *mut ast::Atom) },
            state: ExpansionState::Idle,
            word_idx: 0,
            out: ExpansionOut::default(),
            current_out: Vec::new(),
            meta_offsets: Vec::new(),
            child_script: None,
            cmd_subst_quoted: false,
            assign_ctx,
            has_quoted_empty: false,
            out_exit_code: 0,
        }))
    }

    pub(crate) fn start(_interp: &Interpreter, this: NodeId) -> Yield {
        Yield::Next(this)
    }

    /// Walks the
    /// atom, appending no-IO expansions to `current_out` and yielding to a
    /// child `Script` whenever a `$(...)` is encountered. Re-entered after
    /// `child_done` advances `word_idx`.
    pub(crate) fn next(interp: &Interpreter, this: NodeId) -> Yield {
        'next: loop {
            // Split-borrow: `me` from `nodes`, `vm_args_utf8` from its own
            // field, so `expand_simple_no_io` can expand `$N` without aliasing.
            // R-2: both are `JsCell`-backed; `as_ptr()`/`node_mut()` project
            // disjoint `&mut` from `&Interpreter`.
            let event_loop = interp.event_loop;
            let command_ctx = interp.command_ctx;
            // SAFETY: single-JS-thread; `vm_args_utf8` and `nodes` are
            // disjoint `JsCell` fields (no aliasing between the two borrows).
            let vm_args_utf8 = unsafe { &mut *interp.vm_args_utf8.as_ptr() };
            let me = interp.as_expansion_mut(this);
            match me.state {
                ExpansionState::Idle => {
                    me.state = ExpansionState::Walking;
                    continue;
                }
                ExpansionState::CmdSubst | ExpansionState::ParamWord | ExpansionState::Glob => {
                    // Re-entered while a child is in flight.
                    return Yield::suspended();
                }
                ExpansionState::BraceExpand => {
                    // A `ParamSpace` ended the last argument: nothing is left to expand.
                    if me.current_out.is_empty() && !me.out.buf.is_empty() {
                        me.state = ExpansionState::Done;
                        continue;
                    }
                    Self::do_brace_expand(me);
                    // brace + glob composes: after pushing the literal brace
                    // variants, glob the original pattern. The normal glob path likewise calls
                    // `transition_to_glob_state` directly (see below).
                    if matches!(me.state, ExpansionState::Glob) {
                        return Self::transition_to_glob_state(interp, this);
                    }
                    continue;
                }
                ExpansionState::Done | ExpansionState::Err(_) => break,
                ExpansionState::Walking => {}
            }

            // Copy the BackRef out so `atom` borrows a local, leaving `me`
            // free for the `&mut me.*` field writes below.
            let node = me.node;
            let atom = node.get();
            let atoms_len = atom.atoms_len();
            // Leading `~` in a compound is skipped during the walk and
            // post-processed below.
            let leading_tilde = matches!(atom, ast::Atom::Compound(c)
                if c.atoms.first().is_some_and(|a| matches!(a, ast::SimpleAtom::Tilde)));
            if me.word_idx == 0 && leading_tilde {
                me.word_idx = 1;
            }

            let shell_ptr: *mut ShellExecEnv = me.base.shell;
            while me.word_idx < atoms_len {
                let simple: &ast::SimpleAtom = match atom {
                    ast::Atom::Simple(s) => s,
                    ast::Atom::Compound(c) => &c.atoms[me.word_idx as usize],
                };
                let shell = me.base.shell();
                match Self::expand_simple_no_io(
                    shell,
                    simple,
                    &mut me.current_out,
                    &mut me.meta_offsets,
                    &mut me.has_quoted_empty,
                    true,
                    event_loop,
                    command_ctx,
                    vm_args_utf8,
                ) {
                    Step::Next => {
                        me.word_idx += 1;
                        continue;
                    }
                    Step::CmdSubst => {}
                    Step::Param => {
                        if let Some(word_child) = Self::expand_param(interp, this) {
                            return word_child;
                        }
                        continue 'next;
                    }
                    Step::ParamSpace => {
                        let ast::SimpleAtom::ParamSpace(space) = simple else {
                            unreachable!()
                        };
                        Self::expand_param_space(me, space);
                        me.word_idx += 1;
                        continue;
                    }
                }
                // ── Command substitution: spawn a Script with stdout piped
                //    into a fresh owned buffer.
                let ast::SimpleAtom::CmdSubst(sub) = simple else {
                    unreachable!()
                };
                let quoted = sub.quoted;
                let script_ast: *const ast::Script = &raw const sub.script;
                me.state = ExpansionState::CmdSubst;
                me.cmd_subst_quoted = quoted;

                let io = IO {
                    stdin: interp.root_io().stdin.clone(),
                    stdout: OutKind::Pipe,
                    stderr: interp.root_io().stderr.clone(),
                };
                // SAFETY: `shell_ptr` is a live env owned by the parent state
                // node and outlives this expansion.
                let duped = match unsafe { &mut *shell_ptr }
                    .dupe_for_subshell(&io, ShellExecEnvKind::CmdSubst)
                {
                    Ok(d) => d,
                    Err(e) => {
                        drop(io);
                        interp.throw(ShellErr::new_sys(&e));
                        return Yield::Failed(this);
                    }
                };
                let script = Script::init(interp, duped, script_ast, this, io);
                interp.as_expansion_mut(this).child_script = Some(script);
                return Script::start(interp, script);
            }

            // All sub-atoms expanded — post-process leading tilde then finish.
            if leading_tilde {
                let home = me.base.shell().get_homedir();
                let len_before = me.current_out.len();
                match me.current_out.first() {
                    Some(b'/') | Some(b'\\') => {
                        me.current_out.splice(0..0, home.slice().iter().copied());
                    }
                    Some(_) => me.current_out.insert(0, b'~'),
                    // `~""` expands to $HOME,
                    // but `~$unset` expands to nothing (word is dropped).
                    None if me.has_quoted_empty => {
                        me.current_out.extend_from_slice(home.slice());
                    }
                    None => {}
                }
                // The first two arms prepend; shift the recorded brace
                // metacharacter offsets so they keep pointing at the same
                // bytes. The `extend_from_slice` arm only runs when
                // `current_out` (and therefore `meta_offsets`) is
                // empty, so the shift is a no-op there.
                let prepended = (me.current_out.len() - len_before) as u32;
                if prepended != 0 {
                    for off in &mut me.meta_offsets {
                        *off += prepended;
                    }
                }
                home.deref();
            }
            // Brace expansion
            // first, then glob, else flush as a single word.
            if atom.has_brace_expansion() {
                me.state = ExpansionState::BraceExpand;
                continue;
            }
            if atom.has_glob_expansion() {
                return Self::transition_to_glob_state(interp, this);
            }
            Self::push_current_out(me);
            me.state = ExpansionState::Done;
        }
        let parent = interp.as_expansion(this).base.parent;
        let exit: ExitCode = if matches!(interp.as_expansion(this).state, ExpansionState::Err(_)) {
            1
        } else {
            0
        };
        interp.child_done(parent, this, exit)
    }

    /// Re-tokenize `current_out` (the
    /// fully-expanded word with `{`/`,`/`}` markers preserved by
    /// `expand_simple_no_io`) and push each variant as a separate argv word.
    fn do_brace_expand(me: &mut Expansion) {
        use bun_shell_parser::braces;
        // Only the `{`/`,`/`}` bytes recorded in `meta_offsets` (written
        // by literal BraceBegin/Comma/BraceEnd atoms) are brace-expansion
        // metacharacters. Bytes from Text/Var/cmd-subst expansion — notably JS
        // `${...}` interpolations — are data: backslash-escape them so the
        // brace lexer cannot be steered into emitting extra argv words.
        // (`meta_offsets` also records literal `*`/`**` positions for the glob
        // path; those are inert here since `*` is not in the escape set.)
        let mut escaped: Vec<u8> = Vec::with_capacity(me.current_out.len());
        let mut next_meta = 0usize;
        for (i, &b) in me.current_out.iter().enumerate() {
            if next_meta < me.meta_offsets.len() && me.meta_offsets[next_meta] as usize == i {
                next_meta += 1;
            } else if matches!(b, b'{' | b'}' | b',' | b'\\') {
                escaped.push(b'\\');
            }
            escaped.push(b);
        }
        let brace_str = &escaped[..];
        let mut lexer_output = if bun_core::is_all_ascii(brace_str) {
            bun_core::handle_oom(braces::Lexer::tokenize(brace_str))
        } else {
            bun_core::handle_oom(
                braces::NewLexer::<{ braces::StringEncoding::Wtf8 }>::tokenize(brace_str),
            )
        };
        let count = braces::calculate_expanded_amount(&lexer_output.tokens[..]);
        // Hard cap before preallocation: `calculate_expanded_amount` saturates
        // to `u32::MAX`, so a tiny nested input can otherwise request a huge `Vec`.
        const MAX_BRACE_EXPANSIONS: u32 = 65536;
        if count > MAX_BRACE_EXPANSIONS {
            let msg = format!("too many brace expansions ({count} > {MAX_BRACE_EXPANSIONS})");
            me.state = ExpansionState::Err(Box::new(ShellErr::Custom(msg.into_bytes().into())));
            return;
        }
        let count = count as usize;
        let expanded: Vec<Vec<u8>> = if count == 0 {
            // No `{...}` group formed an expansion (no top-level comma, or
            // unbalanced); emit the word unchanged. `expand` would index
            // `out[0]` of an empty slice here. Keep `current_out` for glob.
            vec![me.current_out.clone()]
        } else {
            let mut expanded: Vec<Vec<u8>> = (0..count).map(|_| Vec::new()).collect();
            let arena = bun_alloc::Arena::new();
            if let Err(e) = braces::expand(
                &arena,
                &mut lexer_output.tokens[..],
                &mut expanded[..],
                lexer_output.contains_nested,
            ) {
                let msg = match e {
                    braces::ParserError::TooManyBraces => "too many braces in brace expansion",
                    // The word has more than `u16::MAX` brace tokens.
                    braces::ParserError::UnexpectedToken => {
                        "unexpected token while expanding braces"
                    }
                    braces::ParserError::OutOfMemory => bun_core::out_of_memory(),
                };
                me.state = ExpansionState::Err(Box::new(ShellErr::Custom(msg.as_bytes().into())));
                return;
            }
            drop(arena);
            expanded
        };

        // Push each variant as its own word; word boundaries are recorded
        // via `bounds`.
        for s in expanded {
            if !me.out.buf.is_empty() {
                me.out.bounds.push(me.out.buf.len() as u32);
            }
            me.out.buf.extend_from_slice(&s);
        }

        let node = me.node;
        let atom = node.get();
        if atom.has_glob_expansion() && Self::has_glob_syntax(me) {
            // brace + glob composes. Keep
            // `current_out` (the original pattern, e.g. `src/*.{ts,tsx}`) so
            // the glob walker brace-expands and globs it; its matches are
            // appended after the literal brace variants already pushed above.
            me.state = ExpansionState::Glob;
        } else {
            me.current_out.clear();
            me.state = ExpansionState::Done;
        }
    }

    /// Build the pattern handed to the glob walker from `current_out`,
    /// neutralizing every glob metacharacter byte that was *not* written by a
    /// literal metacharacter atom (those positions are recorded in
    /// `meta_offsets`). Metacharacters arriving via JS `${...}` interpolation,
    /// `$var`, command substitution, or quoted text are data and must not be
    /// able to broaden the match. Mirrors `do_brace_expand`'s escaping loop,
    /// but the glob matcher has no general backslash-escape that survives
    /// `build_pattern_components` on every platform, so each byte is wrapped
    /// in a single-character class (`[c]`) — or a one-branch brace group for
    /// a component-leading `!` — which the matcher provably treats as that
    /// literal character.
    /// `current_out` itself is not mutated: the no-match error message and the
    /// assignment-position literal fallback keep using the original word.
    fn neutralize_glob_metachars(current_out: &[u8], meta_offsets: &[u32]) -> Vec<u8> {
        let mut pattern: Vec<u8> = Vec::with_capacity(current_out.len());
        let mut next_meta = 0usize;
        for (i, &b) in current_out.iter().enumerate() {
            if next_meta < meta_offsets.len() && meta_offsets[next_meta] as usize == i {
                next_meta += 1;
                pattern.push(b);
                continue;
            }
            match b {
                // `[c]` is a single-character class containing only `c`.
                b'*' | b'?' | b'[' | b']' | b'{' | b'}' | b',' => {
                    pattern.extend_from_slice(&[b'[', b, b']']);
                }
                // The shell has no literal atom for `(` in a word, so here it is always data. It only means something to the
                // matcher after one of these, as the start of an extglob group, and without a group `)` and `|` are plain
                // characters. Any other `(` stays bare, so that `Dropbox (Personal)` is still a literal component, which
                // the walker opens by name: that follows a symlink, and ignores case where the file system does.
                b'(' if matches!(pattern.last(), Some(b'?' | b'*' | b'+' | b'@' | b'!')) => {
                    pattern.extend_from_slice(b"[(]");
                }
                // `[!]`/`[^]` would be a negated class, so `!` cannot use the
                // class wrapper. Only a `!` at the start of a path component
                // (the same split `build_pattern_components` performs) can act
                // as pattern syntax — the matcher's negation loop — so wrap
                // that one in a one-branch brace group whose sole branch is
                // the literal `!`. Every other `!` already matches literally
                // (a `(` after it is wrapped, so it can't start `!(`)
                // and is emitted bare: wrapping each one costs a brace-stack
                // slot per byte, and a run of more than 10 interpolated `!`
                // would overflow the matcher's bounded brace stack and turn
                // the whole word into a spurious no-match.
                b'!' => {
                    let starts_component = pattern
                        .last()
                        .is_none_or(|&prev| bun_core::path_sep::is_sep_native(prev));
                    if starts_component {
                        pattern.extend_from_slice(b"{!}");
                    } else {
                        pattern.push(b'!');
                    }
                }
                // `[\\]` is a class containing an escaped `\`; the 3-byte
                // `[\]` would mis-parse as a class containing an escaped `]`.
                #[cfg(not(windows))]
                b'\\' => pattern.extend_from_slice(b"[\\\\]"),
                // On Windows `\` is a native path separator (same pass-through
                // policy as `/`); wrapping it would let the component splitter
                // cut the synthesized brackets apart.
                _ => pattern.push(b),
            }
        }
        pattern
    }

    /// Whether the word has a `*` from the script. An atom has its glob hint from a `*` of its own, or from the
    /// word of a `${NAME<op>word}`, which adds a `*` only when that word is the result.
    fn has_glob_syntax(me: &Expansion) -> bool {
        let is_glob = |atom: &ast::SimpleAtom| {
            matches!(
                atom,
                ast::SimpleAtom::Asterisk | ast::SimpleAtom::DoubleAsterisk
            )
        };
        let node = me.node;
        match node.get() {
            ast::Atom::Simple(simple) => is_glob(simple),
            ast::Atom::Compound(compound) => {
                compound.atoms.iter().any(is_glob)
                    || me
                        .meta_offsets
                        .iter()
                        .any(|&offset| me.current_out.get(offset as usize) == Some(&b'*'))
            }
        }
    }

    /// Kick off an off-thread
    /// glob walk for the assembled pattern in `current_out`.
    fn transition_to_glob_state(interp: &Interpreter, this: NodeId) -> Yield {
        use crate::shell::dispatch_tasks::ShellGlobTask;
        let pattern: Vec<u8>;
        let cwd: Vec<u8>;
        {
            let me = interp.as_expansion_mut(this);
            // A `ParamSpace` ended the last argument.
            let nothing_left = me.current_out.is_empty() && !me.out.buf.is_empty();
            // The word of a `${NAME<op>word}` is not a pattern in its child `Expansion`: a `*` in it is text there.
            let is_param_word = matches!(interp.node(me.base.parent), Node::Expansion(_));
            if nothing_left || is_param_word || !Self::has_glob_syntax(me) {
                if !nothing_left {
                    Self::push_current_out(me);
                }
                me.state = ExpansionState::Done;
                return Yield::Next(this);
            }
            me.state = ExpansionState::Glob;
            pattern = Self::neutralize_glob_metachars(&me.current_out, &me.meta_offsets);
            cwd = me.base.shell().cwd().to_vec();
        }
        let walker = match bun_glob::BunGlobWalkerZ::init_with_cwd(
            &pattern, &cwd, false, false, false, false, false, None,
        ) {
            Ok(Ok(w)) => w,
            Ok(Err(e)) => {
                interp.as_expansion_mut(this).state =
                    ExpansionState::Err(Box::new(ShellErr::new_sys(&e)));
                return Yield::Next(this);
            }
            Err(e) => {
                interp.as_expansion_mut(this).state = ExpansionState::Err(Box::new(
                    ShellErr::Custom(e.to_string().into_bytes().into()),
                ));
                return Yield::Next(this);
            }
        };
        ShellGlobTask::create_and_schedule(interp, this, walker);
        Yield::suspended()
    }

    /// Appends the no-IO expansion of
    /// one [`ast::SimpleAtom`] to `out`. Returns what is left for the caller:
    /// it spawns a `Script` for a `CmdSubst`.
    fn expand_simple_no_io(
        shell: &ShellExecEnv,
        atom: &ast::SimpleAtom,
        out: &mut Vec<u8>,
        meta_offsets: &mut Vec<u32>,
        has_quoted_empty: &mut bool,
        expand_tilde: bool,
        event_loop: EventLoopHandle,
        command_ctx: *mut bun_options_types::context::ContextData,
        vm_args_utf8: &mut Vec<bun_core::Utf8Bytes<'static>>,
    ) -> Step {
        match atom {
            ast::SimpleAtom::Text(txt) => out.extend_from_slice(txt),
            ast::SimpleAtom::QuotedEmpty => {
                // Sets `has_quoted_empty = true` so an empty word is still pushed
                // as an arg. The flag is *required* — without it Cmd cannot
                // tell `""` (one empty arg) from `$unset` (no arg), since
                // both leave `out.buf` empty.
                *has_quoted_empty = true;
            }
            ast::SimpleAtom::Var(label) => {
                // Spec `expandVar`: shell_env first, then export_env, else "".
                let key = EnvStr::init_slice(label);
                if let Some(v) = shell.shell_env.get(key) {
                    out.extend_from_slice(v.slice());
                    v.deref();
                } else if let Some(v) = shell.export_env.get(EnvStr::init_slice(label)) {
                    out.extend_from_slice(v.slice());
                    v.deref();
                }
            }
            ast::SimpleAtom::VarArgv(int) => {
                // SAFETY: `command_ctx` is the live VM ctx; `vm_args_utf8` borrows it.
                Interpreter::append_var_argv(out, *int, event_loop, command_ctx, vm_args_utf8);
            }
            ast::SimpleAtom::Param(_) => return Step::Param,
            ast::SimpleAtom::ParamSpace(_) => return Step::ParamSpace,
            ast::SimpleAtom::Asterisk => {
                meta_offsets.push(out.len() as u32);
                out.push(b'*');
            }
            ast::SimpleAtom::DoubleAsterisk => {
                // Both bytes must be recorded or `neutralize_glob_metachars`
                // would wrap the second `*`, breaking `**`.
                meta_offsets.push(out.len() as u32);
                meta_offsets.push(out.len() as u32 + 1);
                out.extend_from_slice(b"**");
            }
            ast::SimpleAtom::BraceBegin => {
                meta_offsets.push(out.len() as u32);
                out.push(b'{');
            }
            ast::SimpleAtom::BraceEnd => {
                meta_offsets.push(out.len() as u32);
                out.push(b'}');
            }
            ast::SimpleAtom::Comma => {
                meta_offsets.push(out.len() as u32);
                out.push(b',');
            }
            ast::SimpleAtom::Tilde => {
                if expand_tilde {
                    let home = shell.get_homedir();
                    out.extend_from_slice(home.slice());
                    home.deref();
                } else {
                    out.push(b'~');
                }
            }
            ast::SimpleAtom::CmdSubst(_) => return Step::CmdSubst,
        }
        Step::Next
    }

    /// Expands the `${NAME<op>word}` at `word_idx`. `Some`: its word runs in a child `Expansion` and this node waits.
    #[cold]
    fn expand_param(interp: &Interpreter, this: NodeId) -> Option<Yield> {
        let mut ctx = ParamCtx {
            event_loop: interp.event_loop,
            command_ctx: interp.command_ctx,
            // SAFETY: single-JS-thread; `vm_args_utf8` and `nodes` are
            // disjoint `JsCell` fields (no aliasing between the two borrows).
            vm_args_utf8: unsafe { &mut *interp.vm_args_utf8.as_ptr() },
            earlier_assigns: Self::earlier_prefix_assigns(interp, this),
        };
        let me = interp.as_expansion_mut(this);
        let param = Self::param_at_word_idx(me);
        if !Self::expand_param_value(me, param, &mut ctx) {
            me.word_idx += 1;
            return None;
        }
        if !param.word_has_cmd_subst {
            let splits = !me.assign_ctx;
            if Self::expand_param_word_no_io(me, param, splits, &mut ctx) {
                me.word_idx += 1;
            }
            return None;
        }

        // The value of a `=` and the message of a `?` are one string, like the value of an assignment.
        let one_string = me.assign_ctx
            || matches!(
                param.op.action,
                ast::ParamAction::Assign | ast::ParamAction::Error
            );
        me.state = ExpansionState::ParamWord;
        let shell = me.base.shell;
        let child = Self::init(interp, shell, &raw const param.word, this, one_string);
        interp.as_expansion_mut(this).child_script = Some(child);
        Some(Self::start(interp, child))
    }

    fn param_at_word_idx(me: &Expansion) -> &'static ast::ParamExp {
        let node = me.node;
        let simple = match node.get() {
            ast::Atom::Simple(s) => s,
            ast::Atom::Compound(c) => &c.atoms[me.word_idx as usize],
        };
        let ast::SimpleAtom::Param(param) = simple else {
            unreachable!("the atom at word_idx is a Param")
        };
        param
    }

    /// In the value of a prefix assignment (`A=1 B=${A:-2} cmd`): the assignments before that value.
    fn earlier_prefix_assigns(interp: &Interpreter, this: NodeId) -> &[ast::Assign] {
        let mut parent = interp.as_expansion(this).base.parent;
        // The word of a `${NAME<op>word}` in the value: its parent is the `Expansion` of the value.
        while let Node::Expansion(word_parent) = interp.node(parent) {
            parent = word_parent.base.parent;
        }
        match interp.node(parent) {
            Node::Assigns(assigns) if assigns.ctx == AssignCtx::Cmd => match assigns.state {
                AssignsState::Expanding { idx } => &assigns.node.slice()[..idx as usize],
                _ => &[],
            },
            _ => &[],
        }
    }

    /// Appends the value of the parameter when that is the result. `true`: the result is the word, and nothing was
    /// appended.
    fn expand_param_value(
        me: &mut Expansion,
        param: &ast::ParamExp,
        ctx: &mut ParamCtx<'_>,
    ) -> bool {
        let len = me.current_out.len();
        let set = match param.target {
            ast::ParamTarget::Name(name) => {
                let shell = me.base.shell();
                // `Cmd` keeps prefix assignments in `cmd_local_env`, which also holds those of earlier commands.
                let in_prefix = ctx
                    .earlier_assigns
                    .iter()
                    .any(|assign| EnvMap::is_same_name(assign.label, name));
                let value = in_prefix
                    .then(|| shell.cmd_local_env.get(EnvStr::init_slice(name)))
                    .flatten()
                    .or_else(|| shell.shell_env.get(EnvStr::init_slice(name)))
                    .or_else(|| shell.export_env.get(EnvStr::init_slice(name)));
                match value {
                    Some(value) => {
                        me.current_out.extend_from_slice(value.slice());
                        value.deref();
                        true
                    }
                    None => false,
                }
            }
            // SAFETY: `command_ctx` is the live VM ctx; `vm_args_utf8` borrows it.
            ast::ParamTarget::Positional(int) => Interpreter::append_var_argv(
                &mut me.current_out,
                int,
                ctx.event_loop,
                ctx.command_ctx,
                ctx.vm_args_utf8,
            ),
        };
        let absent = param.op.is_absent(set, me.current_out.len() == len);
        let is_alternate = param.op.action == ast::ParamAction::Alternate;
        if absent == is_alternate {
            // `+` gives nothing for an absent parameter. The other operators give the value of a present one.
            if is_alternate {
                me.current_out.truncate(len);
            }
            return false;
        }
        me.current_out.truncate(len);
        true
    }

    /// Expands the word of a `${NAME<op>word}` that holds no `$(...)` and applies the operator. `splits`: unquoted
    /// whitespace in the word ends an argument. `false`: the expansion failed and `state` is the error.
    fn expand_param_word_no_io(
        me: &mut Expansion,
        param: &ast::ParamExp,
        splits: bool,
        ctx: &mut ParamCtx<'_>,
    ) -> bool {
        debug_assert!(!param.word_has_cmd_subst);
        let ast::Atom::Compound(word) = &param.word else {
            unreachable!("the word of a Param is a compound atom")
        };
        match param.op.action {
            ast::ParamAction::Default | ast::ParamAction::Alternate => {
                Self::expand_word_atoms_no_io(me, word.atoms, splits, ctx)
            }
            ast::ParamAction::Assign | ast::ParamAction::Error => {
                let len = me.current_out.len();
                let meta_len = me.meta_offsets.len();
                let has_quoted_empty = me.has_quoted_empty;
                if !Self::expand_word_atoms_no_io(me, word.atoms, false, ctx) {
                    return false;
                }
                // The word is a value here: a `*` or a `""` in it is not syntax of the argument.
                let value = me.current_out.split_off(len);
                me.meta_offsets.truncate(meta_len);
                me.has_quoted_empty = has_quoted_empty;
                Self::apply_param_word(me, param, value)
            }
        }
    }

    fn expand_word_atoms_no_io(
        me: &mut Expansion,
        atoms: &[ast::SimpleAtom],
        splits: bool,
        ctx: &mut ParamCtx<'_>,
    ) -> bool {
        for atom in atoms {
            match atom {
                ast::SimpleAtom::Param(inner) => {
                    if Self::expand_param_value(me, inner, ctx)
                        && !Self::expand_param_word_no_io(me, inner, splits, ctx)
                    {
                        return false;
                    }
                }
                ast::SimpleAtom::ParamSpace(_) if splits => {
                    if !me.current_out.is_empty() {
                        Self::push_current_out(me);
                    }
                }
                ast::SimpleAtom::ParamSpace(space) => me.current_out.extend_from_slice(space),
                _ => {
                    let shell = me.base.shell();
                    let _ = Self::expand_simple_no_io(
                        shell,
                        atom,
                        &mut me.current_out,
                        &mut me.meta_offsets,
                        &mut me.has_quoted_empty,
                        true,
                        ctx.event_loop,
                        ctx.command_ctx,
                        ctx.vm_args_utf8,
                    );
                }
            }
        }
        true
    }

    /// Unquoted whitespace in the word of a `${NAME<op>word}`: it ends the argument where arguments are split.
    fn expand_param_space(me: &mut Expansion, space: &[u8]) {
        if me.assign_ctx {
            me.current_out.extend_from_slice(space);
        } else if !me.current_out.is_empty() {
            Self::push_current_out(me);
        }
    }

    /// Ends a `=` or a `?` form whose word expanded to `word`. `false`: `state` is an error.
    fn apply_param_word(me: &mut Expansion, param: &ast::ParamExp, word: Vec<u8>) -> bool {
        use std::io::Write as _;
        let mut message: Vec<u8> = Vec::new();
        match (param.op.action, param.target) {
            (ast::ParamAction::Assign, ast::ParamTarget::Name(name)) => {
                me.current_out.extend_from_slice(&word);
                let value = EnvStr::init_ref_counted(word.into_boxed_slice());
                me.base
                    .shell_mut()
                    .assign_var(EnvStr::init_slice(name), value, AssignCtx::Shell);
                value.deref();
                return true;
            }
            (ast::ParamAction::Assign, ast::ParamTarget::Positional(int)) => {
                // A write to a `Vec` cannot fail.
                let _ = write!(message, "${int}: cannot assign in this way");
            }
            (_, target) => {
                match target {
                    ast::ParamTarget::Name(name) => message.extend_from_slice(name),
                    ast::ParamTarget::Positional(int) => {
                        let _ = write!(message, "{int}");
                    }
                }
                message.extend_from_slice(b": ");
                if !word.is_empty() {
                    message.extend_from_slice(&word);
                } else if param.op.colon {
                    message.extend_from_slice(b"parameter null or not set");
                } else {
                    message.extend_from_slice(b"parameter not set");
                }
            }
        }
        me.state = ExpansionState::Err(Box::new(ShellErr::Custom(message.into_boxed_slice())));
        false
    }

    /// The child `Expansion` of the word of the `${NAME<op>word}` at `word_idx` is done.
    #[cold]
    fn param_word_done(
        interp: &Interpreter,
        this: NodeId,
        child: NodeId,
        exit_code: ExitCode,
    ) -> Yield {
        if exit_code != 0 {
            let err = Self::take_err(interp, child);
            interp.deinit_node(child);
            let me = interp.as_expansion_mut(this);
            me.child_script = None;
            let Some(err) = err else {
                // The script failed in a `$(...)` of the word.
                me.state = ExpansionState::Done;
                let parent = me.base.parent;
                return interp.child_done(parent, this, 1);
            };
            me.state = ExpansionState::Err(Box::new(err));
            return Yield::Next(this);
        }

        let out = Self::take_out(interp, child);
        interp.deinit_node(child);
        let me = interp.as_expansion_mut(this);
        me.child_script = None;
        me.state = ExpansionState::Walking;
        let param = Self::param_at_word_idx(me);
        let done = match param.op.action {
            ast::ParamAction::Default | ast::ParamAction::Alternate => {
                Self::append_param_word_fields(me, param, &out);
                true
            }
            ast::ParamAction::Assign | ast::ParamAction::Error => {
                Self::apply_param_word(me, param, out.buf)
            }
        };
        if done {
            me.word_idx += 1;
        }
        Yield::Next(this)
    }

    /// Adds the arguments that the word of a `${NAME<op>word}` expanded to in its child `Expansion`.
    fn append_param_word_fields(me: &mut Expansion, param: &ast::ParamExp, out: &ExpansionOut) {
        let ast::Atom::Compound(word) = &param.word else {
            unreachable!("the word of a Param is a compound atom")
        };
        // The child drops whitespace at the start of the word and ends its last argument at whitespace at the end.
        let ends_argument = |atom: Option<&ast::SimpleAtom>| {
            !me.assign_ctx && matches!(atom, Some(ast::SimpleAtom::ParamSpace(_)))
        };
        let (ends_before, ends_after) = (
            ends_argument(word.atoms.first()),
            ends_argument(word.atoms.last()),
        );
        if ends_before && !me.current_out.is_empty() {
            Self::push_current_out(me);
        }
        let mut start = 0usize;
        for &end in &out.bounds {
            me.current_out
                .extend_from_slice(&out.buf[start..end as usize]);
            Self::push_current_out(me);
            start = end as usize;
        }
        me.current_out.extend_from_slice(&out.buf[start..]);
        me.has_quoted_empty |= out.has_quoted_empty;
        if ends_after && !me.current_out.is_empty() {
            Self::push_current_out(me);
        }
    }

    /// Flush `current_out` into `out`
    /// as the next argv word. The word boundary is recorded as the *previous*
    /// end-offset so the consumer's `[prev..bound]` slicing reconstructs each
    /// word and the trailing `[prev..]` slice yields the final one.
    fn push_current_out(me: &mut Expansion) {
        if !me.out.buf.is_empty() {
            me.out.bounds.push(me.out.buf.len() as u32);
        }
        me.out.buf.append(&mut me.current_out);
        me.meta_offsets.clear();
    }

    /// Newlines→spaces, trim, then split on whitespace runs into separate
    /// argv words.
    fn post_subshell_expansion(me: &mut Expansion, mut stdout: Vec<u8>) {
        // Strip a single trailing newline, then convert remaining newlines
        // to spaces.
        if stdout.last() == Some(&b'\n') {
            stdout.pop();
        }
        for b in stdout.iter_mut() {
            if *b == b'\n' {
                *b = b' ';
            }
        }
        // Trim leading/trailing whitespace.
        let s: &[u8] = {
            let mut lo = 0usize;
            let mut hi = stdout.len();
            while lo < hi && matches!(stdout[lo], b' ' | b'\n' | b'\r' | b'\t') {
                lo += 1;
            }
            while hi > lo && matches!(stdout[hi - 1], b' ' | b'\n' | b'\r' | b'\t') {
                hi -= 1;
            }
            &stdout[lo..hi]
        };
        if s.is_empty() {
            return;
        }
        // Split on runs of spaces — each run is a word boundary.
        let mut prev_ws = false;
        let mut a = 0usize;
        for (i, &c) in s.iter().enumerate() {
            if prev_ws {
                if c != b' ' {
                    a = i;
                    prev_ws = false;
                }
                continue;
            }
            if c == b' ' {
                prev_ws = true;
                me.current_out.extend_from_slice(&s[a..i]);
                Self::push_current_out(me);
            }
        }
        me.current_out.extend_from_slice(&s[a..]);
    }

    pub(crate) fn child_done(
        interp: &Interpreter,
        this: NodeId,
        child: NodeId,
        exit_code: ExitCode,
    ) -> Yield {
        if matches!(interp.as_expansion(this).state, ExpansionState::ParamWord) {
            return Self::param_word_done(interp, this, child, exit_code);
        }
        // Child is a Script (command substitution). Its captured stdout lives
        // in the duped `ShellExecEnv` it owns; read it before deinit.
        debug_assert!(matches!(interp.node(child).kind(), StateKind::Script));
        if interp.failed() {
            // The script failed: the rest of the word is not expanded.
            {
                let me = interp.as_expansion_mut(this);
                me.state = ExpansionState::Done;
                me.child_script = None;
            }
            interp.deinit_node(child);
            let parent = interp.as_expansion(this).base.parent;
            return interp.child_done(parent, this, 1);
        }
        // SAFETY: single trampoline frame; the child script's env (and its
        // parent buffer in the `Borrowed` case) has no other live borrow.
        let mut stdout = unsafe {
            interp
                .as_script_mut(child)
                .base
                .shell_mut()
                .buffered_stdout_mut()
        }
        .clone();
        // Like bash and dash, drop NUL bytes before trimming and field splitting.
        let had_nul = bun_core::strings::contains_char(&stdout, 0);
        if had_nul {
            stdout.retain(|&b| b != 0);
        }

        // Propagate the exit code if the *whole* atom was a single `$(...)`
        // (so `$(false)` as argv0 fails the command).
        let sole_cmd_subst = matches!(
            interp.as_expansion(this).node.get(),
            ast::Atom::Simple(ast::SimpleAtom::CmdSubst(_))
        );

        let no_split = {
            let me = interp.as_expansion(this);
            me.cmd_subst_quoted || me.assign_ctx
        };
        {
            let me = interp.as_expansion_mut(this);
            if exit_code != 0 && sole_cmd_subst {
                me.out_exit_code = exit_code;
            }
            if no_split {
                let mut hi = stdout.len();
                while hi > 0 && matches!(stdout[hi - 1], b' ' | b'\n' | b'\r' | b'\t') {
                    hi -= 1;
                }
                me.current_out.extend_from_slice(&stdout[..hi]);
                // A quoted substitution that printed only NUL bytes expands to one empty word.
                if had_nul && me.cmd_subst_quoted {
                    me.has_quoted_empty = true;
                }
            } else {
                Self::post_subshell_expansion(me, stdout);
            }
            me.word_idx += 1;
            me.state = ExpansionState::Walking;
            me.child_script = None;
        }
        interp.deinit_node(child);
        Yield::Next(this)
    }

    /// Main-thread re-entry for the
    /// off-thread glob walker — splice each match as a separate word into
    /// `out` then resume the atom-walk trampoline.
    pub(crate) fn on_glob_walk_done(
        interp: &Interpreter,
        this: NodeId,
        result: Vec<Vec<u8>>,
        err: Option<crate::shell::dispatch_tasks::ShellGlobErr>,
    ) {
        use crate::shell::dispatch_tasks::ShellGlobErr;
        log!("Expansion {} onGlobWalkDone", this);
        let walk_err = match err {
            Some(ShellGlobErr::Syscall(e))
                if matches!(e.get_errno(), bun_sys::E::ENOENT | bun_sys::E::ENOTDIR) =>
            {
                log!("Expansion {} glob walk failed: {}", this, e);
                None
            }
            other => other,
        };

        if result.is_empty() || walk_err.is_some() {
            // In variable assignments a no-match glob
            // expands to the literal pattern; otherwise it's an error.
            let parent = interp.as_expansion(this).base.parent;
            let in_assign = matches!(interp.node(parent).kind(), StateKind::Assign)
                || matches!(
                    interp.node(parent),
                    Node::Cmd(c) if matches!(
                        c.state,
                        crate::shell::states::cmd::CmdState::ExpandingAssigns
                    )
                );
            let me = interp.as_expansion_mut(this);
            if in_assign {
                Self::push_current_out(me);
                me.state = ExpansionState::Done;
            } else if let Some(err) = walk_err {
                let shell_err = match err {
                    ShellGlobErr::Syscall(e) => ShellErr::new_sys(&e),
                    ShellGlobErr::Unknown(e) => ShellErr::Custom(e.to_string().into_bytes().into()),
                };
                me.state = ExpansionState::Err(Box::new(shell_err));
            } else {
                let msg = format!("no matches found: {}", bstr::BStr::new(&me.current_out));
                me.state = ExpansionState::Err(Box::new(ShellErr::Custom(msg.into_bytes().into())));
            }
            Yield::Next(this).run(interp);
            return;
        }

        {
            let me = interp.as_expansion_mut(this);
            // Push each match as its own argv word. The
            // walker arena owns the strings, so they were `to_vec`'d already.
            for entry in result {
                if !me.out.buf.is_empty() {
                    me.out.bounds.push(me.out.buf.len() as u32);
                }
                me.out.buf.extend_from_slice(&entry);
            }
            me.state = ExpansionState::Done;
        }
        Yield::Next(this).run(interp);
    }

    /// Take the error out of `state == Err(_)` (called by the parent on
    /// `child_done(_, 1)` to print it). Leaves `state == Done`.
    pub(crate) fn take_err(interp: &Interpreter, this: NodeId) -> Option<ShellErr> {
        let me = interp.as_expansion_mut(this);
        match core::mem::replace(&mut me.state, ExpansionState::Done) {
            ExpansionState::Err(e) => Some(*e),
            other => {
                me.state = other;
                None
            }
        }
    }

    pub(crate) fn deinit(interp: &Interpreter, this: NodeId) {
        log!("Expansion {} deinit", this);
        let child = interp.as_expansion_mut(this).child_script.take();
        if let Some(c) = child {
            interp.deinit_node(c);
        }
        let me = interp.as_expansion_mut(this);
        me.out.buf.clear();
        me.out.bounds.clear();
        me.current_out.clear();
    }

    /// Take the expanded output (called by the parent after `child_done`).
    pub(crate) fn take_out(interp: &Interpreter, this: NodeId) -> ExpansionOut {
        let me = interp.as_expansion_mut(this);
        let mut out = core::mem::take(&mut me.out);
        out.out_exit_code = me.out_exit_code;
        out.has_quoted_empty = me.has_quoted_empty;
        out
    }
}
