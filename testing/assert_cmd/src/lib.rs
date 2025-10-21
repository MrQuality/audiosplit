use std::env;
use std::ffi::OsStr;
use std::io;
use std::path::PathBuf;
use std::process::{Command as ProcessCommand, Output};

pub struct Command {
    inner: ProcessCommand,
}

impl Command {
    pub fn cargo_bin(name: &str) -> io::Result<Self> {
        let var = format!("CARGO_BIN_EXE_{}", name);
        let bin = env::var_os(&var)
            .or_else(|| fallback_bin(name).map(|p| p.into_os_string()))
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "binary path not exported"))?;
        Ok(Self {
            inner: ProcessCommand::new(bin),
        })
    }

    pub fn arg<S: AsRef<OsStr>>(&mut self, arg: S) -> &mut Self {
        self.inner.arg(arg);
        self
    }

    pub fn args<I, S>(&mut self, args: I) -> &mut Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        self.inner.args(args);
        self
    }

    pub fn assert(&mut self) -> Assert {
        let output = self
            .inner
            .output()
            .expect("failed to execute command for assertion");
        Assert { output }
    }
}

fn fallback_bin(name: &str) -> Option<PathBuf> {
    let manifest_dir = env::var_os("CARGO_MANIFEST_DIR")?;
    let mut path = PathBuf::from(manifest_dir);
    path.pop();
    path.push("target");
    let profile = env::var_os("PROFILE").unwrap_or_else(|| "debug".into());
    path.push(profile);
    #[cfg_attr(not(windows), allow(unused_mut))]
    let mut bin_path = path.join(name);
    #[cfg(windows)]
    {
        bin_path.set_extension("exe");
    }
    if bin_path.exists() {
        Some(bin_path)
    } else {
        None
    }
}

pub struct Assert {
    output: Output,
}

impl Assert {
    pub fn success(self) -> Self {
        assert!(
            self.output.status.success(),
            "expected success but got status: {}\nstderr: {}",
            self.output.status,
            String::from_utf8_lossy(&self.output.stderr)
        );
        self
    }

    pub fn failure(self) -> Self {
        assert!(
            !self.output.status.success(),
            "expected failure but command succeeded"
        );
        self
    }

    pub fn stderr_contains(self, needle: &str) -> Self {
        let stderr = String::from_utf8_lossy(&self.output.stderr);
        assert!(
            stderr.contains(needle),
            "stderr did not contain '{needle}'.\nActual stderr: {stderr}"
        );
        self
    }
}
