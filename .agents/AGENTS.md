# Orca Framework - Agent Instructions

Welcome to the Orca project! You are an AI agent assisting the user (the "pilot") in building a lightweight, modular, and fast Machine Learning framework from scratch. The core is written in **Rust** for performance and memory safety, while the frontend is in **Python** (via PyO3/Maturin) to mimic the PyTorch API.

## 🎯 Visi Proyek
**"Simple by default. Powerful when needed."**
- Orca harus mudah dipahami secara arsitektur (modular).
- Tidak mengorbankan performa (siap menggunakan GPU Backend).
- Menghindari *spaghetti code* dan *circular dependencies*.

## 🏗️ Struktur Repositori
- `orca-core/`: Mendefinisikan `DType`, `Device`, `Shape`, dan `OrcaError`.
- `orca-tensor/`: Representasi `Tensor<B: Backend>` dan operasi tensor dasar.
- `orca-autograd/`: Mesin autodiff reverse-mode berbasis tape.
- `orca-backend-cpu/`: Backend CPU referensi.
- `orca-backend-gpu/`: Backend GPU berbasis `wgpu`.
- `orca-distributed/`: Kolektif TCP dasar untuk `all_reduce`.
- `orca-serialize/`: Helper serialisasi tensor.
- `orca-python/`: Binding Rust ke Python via PyO3.
- `python/orca/`: Frontend Python (`nn`, `optim`, `data`, `autocast`).

## 🚦 Status Proyek
**Current shipped surface**
- Core tensor stack: `orca-core`, `orca-tensor`, `orca-autograd`.
- Execution backends: `orca-backend-cpu`, `orca-backend-gpu`.
- Python bindings: `orca-python` plus `python/orca/*`.
- Distributed baseline: `orca-distributed` TCP `all_reduce`.
- Mixed precision and quantization helpers are present on the Python side.

**Roadmap**
- Ikuti `docs/foundation/05-ROADMAP.md` untuk fase berikutnya.
- Jangan menganggap fase roadmap sebagai sudah selesai kecuali ada bukti kode, test, dan dokumentasi yang sinkron.

## ⚠️ Rules & Coding Standards for Agents
1. **Pahami Dulu, Eksekusi Kemudian**: Jangan hanya menjadi 'Yes-Man'. Teliti apakah instruksi *user* secara teknis solid. Berikan spekulasi, data, dan alasan teknis sebelum menulis *code*.
2. **Error Handling**: JANGAN PERNAH gunakan `.unwrap()` atau `panic!` pada *library code* (`src/`), gunakan *proper Error propagation* (`Result`, `OrcaError`). `.unwrap()` hanya boleh di *script testing* atau *Autograd unwrap* jika tipe data sudah dipastikan benar 100%.
3. **No Circular Dependencies**: Jaga hirarki crate. `orca-core` tidak boleh bergantung pada yang lain. `orca-autograd` hanya bergantung pada `orca-tensor` dan `orca-core`.
4. **PyO3 Signatures**: Pastikan kompatibel dengan versi terbaru PyO3 0.21+ (cth: `m: &Bound<'_, PyModule>`).
5. **Workflow Build**: Gunakan `maturin develop` dari root *workspace* (ingat `.venv\Scripts\Activate.ps1`) untuk melakukan *build* ke Python environment. Selalu tes *code* dengan menjalankan ulang _script_ training.

Bacalah file ini dengan seksama setiap kali kamu mengambil alih sesi!
