use std::env;
use std::ffi::OsStr;
use std::io;
use std::process::{Command as ProcessCommand, Output};

pub struct Command {
    inner: ProcessCommand,
}

impl Command {
    pub fn cargo_bin(name: &str) -> io::Result<Self> {
        let var = format!("CARGO_BIN_EXE_{}", name);
        let bin = env::var_os(var)
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
