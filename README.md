# Panggung Mitologi Nusantara

Program interaktif berbasis kamera: tokoh mitologi Nusantara muncul di dalam kubus 3D
yang dikendalikan dengan gerakan tangan. Dibuat dengan Python, OpenCV, dan MediaPipe Hand Landmarker.

**Tokoh (berurutan):** Hanoman, Naga Besukih, Nyi Blorong, Garuda, Banaspati, Kuyang, Nyi Roro Kidul, Rangda, Genderuwo, Ahool, Kuda Sembrani.

## Alur gerakan

| Tangan | Gerakan | Hasil |
|---|---|---|
| A | Tempelkan jempol & telunjuk (tahan sebentar), lalu buka | Kubus kosong keluar ke samping dari celah jempol-telunjuk. Kubus tetap ada selama jari terbuka; lebar bukaan = ukuran kubus. Telapak terbuka atau kepalan biasa TIDAK memunculkan kubus |
| A | Tempelkan lagi jempol & telunjuk | Kubus hilang |
| B | Kepalkan | Asap tebal memenuhi kubus, warnanya sesuai tokoh yang akan muncul |
| B | Buka kepalan | Tokoh baru muncul: titik-titik node berkumpul, lalu menjadi gambar tokoh 4D (bervolume, cahaya bergerak, kilau partikel) |
| B | Kepal-buka lagi | Tokoh berikutnya (berurutan sesuai `tokoh.json`) |

## Isi folder

```
mitologi_hands_project/
├── mitologi_hands.py      program utama (jalankan ini)
├── image_nodes.py         pengubah gambar menjadi node
├── gerak.py               gerakan tokoh (sayap, api, rambut, dll.)
├── tokoh.json             daftar tokoh: gambar, nama, keterangan, urutan
├── gambar/                gambar tokoh
├── .cache/                hasil olahan gambar (dibuat otomatis, boleh dihapus)
├── requirements.txt
└── hand_landmarker.task   model tangan MediaPipe (salin dari proyek lain)
```

## Menjalankan di VS Code (Mac)

```bash
git clone https://github.com/des1susanti/panggung-mitologi-nusantara.git
cd panggung-mitologi-nusantara
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python mitologi_hands.py
```

File model `hand_landmarker.task` diunduh otomatis saat program pertama kali dijalankan.
Di terminal harus muncul `Kamera: OK` dan `Pelacak tangan: OK`.

## Menambah atau mengganti tokoh

1. Taruh gambar baru (PNG/JPG, sebaiknya latar polos) di folder `gambar/`.
2. Tambahkan satu baris di `tokoh.json`, misalnya:
   `{"gambar": "11_barong.png", "nama": "BARONG", "keterangan": "Pelindung dari Bali"}`
3. Urutan tampil mengikuti urutan baris di `tokoh.json`. Hapus baris untuk membuang tokoh.

Pengaturan tambahan per tokoh (opsional) bila latarnya ikut terbawa:
- `"potong": [kiri, atas, kanan, bawah]` memotong tepi gambar dalam persen (mis. buang watermark di bawah).
- `"mode": "outline"` untuk tokoh berwarna putih di latar putih yang punya garis tepi.
- `"batas": [kiri, atas, kanan, bawah]` jarak bingkai pemisah latar (pecahan, mis. 0.05).
- `"hapus_terang_bawah": 0.6` membuang cahaya terang di bagian bawah gambar.
- `"area": [[x,y], ...]` poligon kira-kira letak tokoh (pecahan 0-1); di luar poligon pasti latar.
- `"pasti": [[x,y], ...]` poligon bagian yang pasti termasuk tokoh (mis. rambut gelap, mahkota); boleh beberapa poligon.
- `"hapus_langit": true` membuang sisa warna langit biru/awan.
- `"warna_asap": "#RRGGBB"` warna asap sebelum tokoh itu muncul (tanpa ini, diambil dari warna gambar).

Kepekaan jepit jempol-telunjuk bisa diatur di `PINCH_CLOSE` dan `PINCH_OPEN` dalam `mitologi_hands.py`.

## Gerakan tokoh

Setiap tokoh bergerak sendiri dari satu gambar: sayap Garuda dan Ahool mengepak,
Naga meliuk, api Banaspati dan surai Kuda Sembrani berkobar, rambut Kuyang berkibar, dsb.
Pengaturannya ada di bagian `"gerak"` setiap tokoh di `tokoh.json`
(jenis: `napas`, `goyang`, `putar`, `liuk`, `kibar`, `api`; penjelasan lengkap di `gerak.py`).
Perbesar `"kuat"` agar gerakan lebih besar, `"laju"` agar lebih cepat.
Hapus bagian `"gerak"` kalau tokoh ingin diam.

## Tombol

| Tombol | Fungsi |
|---|---|
| H | Sembunyikan/tampilkan titik tangan |
| I | Tampilkan jumlah tangan & fps |
| F | Ganti layar penuh / jendela (program terbuka layar penuh) |
| Q / Esc | Keluar |

Buka dalam jendela biasa: `python mitologi_hands.py --jendela`.

Coba tanpa kamera: `python mitologi_hands.py --demo`, lalu tekan **spasi** (kepal/buka tangan B bergantian).

## Catatan gambar

Gambar tokoh di folder `gambar/` diambil dari berbagai sumber di internet dan hak ciptanya
milik pembuatnya masing-masing. Gambar dipakai untuk keperluan belajar dan tidak untuk
dikomersialkan.
