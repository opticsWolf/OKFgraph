//! Test helper: hold the OKFgraph inter-process write lock for N ms.
use std::fs::OpenOptions;
use std::io::Write;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: lock_hold <lock-path> <ms>");
    let millis: u64 = args
        .next()
        .expect("usage: lock_hold <lock-path> <ms>")
        .parse()
        .expect("milliseconds");
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&path)
        .expect("open lock file");
    let mut lock = fd_lock::RwLock::new(file);
    let _guard = lock.write().expect("acquire lock");
    println!("locked");
    std::io::stdout().flush().ok();
    std::thread::sleep(std::time::Duration::from_millis(millis));
}
