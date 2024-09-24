import base64
from collections import defaultdict
import json
import os
import tempfile
import cv2
import flet as ft
import threading
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
from io import BytesIO
from measurement import measure
from calibration import calibration

# from process.coin_detect import detect_onnx
from onnx_rt import build_model, process_frame
import sys
from flet.matplotlib_chart import MatplotlibChart

matplotlib.use("svg")


stop_flags = {0: threading.Event(), 1: threading.Event()}
import time
from datetime import datetime

now = datetime.now()
formatted_date = now.strftime("%d %B %Y, %H:%M")


def write_json(new_data, filename="results.json"):
    with open(filename, "r+") as file:
        file_data = json.load(file)
        file_data["results"].append({"data": new_data, "date": formatted_date})

        file.seek(0)
        json.dump(file_data, file, indent=4)
        file.truncate()


selected_index = 0
measurement_result = [0, 0, 0, 0, 0]


def load_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def main(page: ft.Page):
    page.window_width = 1024
    page.window_height = 600

    def change_content(index):
        content_container.controls.clear()
        if index == 0:
            content_container.controls.append(home())
        elif index == 1:
            content_container.controls.append(kalibrasi())
        elif index == 2:
            content_container.controls.append(pengukuran())
        elif index == 3:
            content_container.controls.append(riwayat())
        elif index == 4:
            content_container.controls.append(petunjuk())
        elif index == 11:
            content_container.controls.append(pengukuran2())
        elif index == 12:
            content_container.controls.append(result())
        page.update()

    def update_frame(image_file, img_control):
        img_control.src_base64 = image_file.decode("utf-8")
        page.update(img_control)

    def start_video(vc_index, img_control):
        is_cuda = len(sys.argv) > 1 and sys.argv[1] == "cuda"
        net = build_model(is_cuda, "model/best.onnx")
        vc = cv2.VideoCapture(vc_index)

        if not vc.isOpened():
            print(f"Failed to open Camera {vc_index}")
            return

        while not stop_flags[vc_index].is_set():
            rval, frame = vc.read()
            if not rval:
                print(f"Camera {vc_index} has stopped delivering frames.")
                break

            if selected_index == 1:
                frame = process_frame(frame, net)

            _, im_arr = cv2.imencode(".jpg", frame)
            im_bytes = im_arr.tobytes()
            im_b64 = base64.b64encode(im_bytes)
            update_frame(im_b64, img_control)

            # Small delay to prevent CPU overload
            cv2.waitKey(1)

        vc.release()

    def stop_all_cameras(e):
        for key in stop_flags.keys():
            stop_flags[key].set()
        print("Cameras stopped.")

    def start_all_cameras(e):
        for key in stop_flags.keys():
            stop_flags[key].clear()

        threading.Thread(target=start_video, args=(0, img)).start()
        threading.Thread(target=start_video, args=(1, img1)).start()

    def start_camera_pengukuran(e):
        for key in stop_flags.keys():
            stop_flags[key].clear()
        threading.Thread(target=start_video, args=(0, img2)).start()
        threading.Thread(target=start_video, args=(1, img3)).start()

    def capture_image(vc_index, filename):
        vc = cv2.VideoCapture(vc_index)

        if not vc.isOpened():
            print(f"Failed to open Camera {vc_index} for capturing image.")
            return

        rval, frame = vc.read()

        if rval:
            cv2.imwrite(filename, frame)
            print(f"Image saved to {filename}")
        else:
            print(f"Failed to capture image from Camera {vc_index}.")

        vc.release()

    def stop_camera_pengukuran1(e):
        stop_flags[0].clear()  # Ensure the flag is cleared before capturing
        stop_flags[0].set()

    def stop_camera_pengukuran2(e):
        stop_flags[1].clear()  # Ensure the flag is cleared before capturing
        stop_flags[1].set()

    def capture_pengukuran1(e):
        capture_image(0, "pengukuran1.jpg")

        change_content(11)

    def capture_pengukuran2(e):  # Ensure the flag is cleared before capturing
        capture_image(1, "pengukuran2.jpg")
        time.sleep(0.5)
        global measurement_result
        result = measure("pengukuran1.jpg", "pengukuran2.jpg")
        print(result)
        measurement_result = result

        if result:
            change_content(12)
        else:
            print("There was an issue with the measurement.")

    def capture_all_images(e):
        capture_image(0, "calibration1.jpg")
        capture_image(1, "calibration2.jpg")

        coef = calibration("calibration1.jpg", "calibration2.jpg")
        dlg_success = ft.AlertDialog(
            title=ft.Text("Kalibrasi berhasil"),
            content=ft.Text("Pergi ke halaman pengukuran untuk melakukan pengukuran!"),
            actions_alignment=ft.MainAxisAlignment.END,
        )
        dlg_error = ft.AlertDialog(
            title=ft.Text("Kalibrasi gagal"),
            content=ft.Text("Silakan melakukan kalibrasi ulang!"),
            actions_alignment=ft.MainAxisAlignment.END,
        )

        if coef.all() is not None:
            page.open(dlg_success)
        else:
            page.open(dlg_error)

    # Membuat direktori untuk menyimpan file sementara jika belum ada
    result_data = load_json("results.json")
    assets_dir = os.path.join(os.getcwd(), "assets")
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)

    def download_action(e, i):
        file_data = download(i)
        with tempfile.NamedTemporaryFile(
            delete=False, dir=assets_dir, suffix=".xlsx"
        ) as tmp:
            with open(tmp.name, "wb") as f:
                f.write(file_data.getvalue())
            # Menggunakan launch_url untuk membuka file
            e.page.launch_url(f"file://{tmp.name}", web_window_name="_self")

    def download(i):
        if i == "all":
            # Mengambil semua data
            data_list = [entry["data"] for entry in result_data["results"]]
            dates = [entry["date"] for entry in result_data["results"]]
            df = pd.DataFrame(data_list)
            df["date"] = dates
        else:
            i = int(i)
            data_entry = result_data["results"][i]["data"]
            date = result_data["results"][i]["date"]
            df = pd.DataFrame([data_entry])
            df["date"] = [date]
        file_data = BytesIO()
        df.to_excel(file_data, index=False, engine="openpyxl")
        file_data.seek(0)
        return file_data

    def petunjuk():

        return ft.Container(
            content=ft.Column(
                [
                    ft.Column(
                        [
                            # Petunjuk Pengukuran
                            # Sebelum melakukan pengukuran, perlu dilakukan kalibrasi terlebih dahulu. Kalibrasi cukup dilakukan satu kali jika posisi kamera tetap. Untuk melakukan kalibrasi beberapa hal berikut perlu dilakukan:
                            # 1. Persiapkan dua uang koin 500 dan background yang bersih dari benda-benda lain agar koin terdeteksi dengan baik
                            # 2. Klik menu kalibrasi pada menu kemudian mulai kamera 3. Letakkan koin 500 untuk kamera atas dan samping
                            # Sesuaikan posisi koin agar sejajar dengan permukaan yang diukur
                            # 5. Simpan hasil kalibrasi dengan klik tombol "Simpan hasil kalibrasi
                            # Cek Video Petunjuk Kalibrasi
                            # Untuk memulai pengukuran.
                            # 1. Pilih menu pengukuran lalu mulai kamera 2. Pertama akan terlihat kamera atas
                            # 3. Klik lanjutkan untuk melanjutkan ke kamera samping Klik mulai pengkuran untuk memulai proses pengukuran
                            # 5. Hasil pengukuran akan ditampilkan secara langsung dan bisa disimpan pada menu riwayat
                            # Jika
                            # hasil kurang akurat ulangi langkah-langkah tadi dengan memperhatikan beberapa hal berikut.
                            # 1. Perhatikan posisi bayi agar tidak miring dilihat dari kamera.
                            # 2. Pastikan tangan dan kaki bayi tidak menutupi bagian tubuh yang lain 3. Jika posisi bayi dirasa sudah pas, pengukuran siap dilakukan
                            # Cak Video Petunjuk Kalibrasi
                            ft.Text("Petunjuk Pengukuran", weight=ft.FontWeight.BOLD),
                            ft.Text(
                                "Sebelum melakukan pengukuran perlu dilakukan kalibrasi terlebih dahulu. Kalibrasi cukup dilakukan satu kali jika posisi kamera tetap. Untuk melakukan kalibrasi beberapa hal berikut perlu dilakukan:",
                                size=14,
                            ),
                            ft.Column(
                                [
                                    ft.Text(
                                        "1. Persiapkan dua uang koin 500 dan background atau latar belakang yang bersih dari benda-benda lain agar koin terdeteksi dengan baik",
                                        size=14,
                                    ),
                                    ft.Text(
                                        "2. Klik menu kalibrasi pada menu di samping, kemudian mulai kamera ",
                                        size=14,
                                    ),
                                    ft.Text(
                                        "3. Letakkan koin 500 untuk kamera atas dan samping",
                                        size=14,
                                    ),
                                    ft.Text(
                                        "4. Sesuaikan posisi koin agar sejajar dengan permukaan yang diukur",
                                        size=14,
                                    ),
                                    ft.Text(
                                        "5. Stop kamera apabila koin sudah sejajar dan terdeteksi (ada persegi biru di koin)",
                                        size=14,
                                    ),
                                    ft.Text(
                                        '6. Proses dan simpan hasil kalibrasi dengan klik tombol "Proses kalibrasi"',
                                        size=14,
                                    ),
                                    ft.Text(
                                "Berikut merupakan video petunjuk kalibrasi",
                            ),
                                    ft.Container(
                                        ft.Video(
                                            playlist=ft.VideoMedia(
                                                "images/video2.mp4"
                                            ),
                                            autoplay=False,
                                            aspect_ratio=16 / 9,
                                            expand=True,
                                            fill_color=ft.colors.TRANSPARENT,
                                            on_loaded=lambda e: print("Loaded"),
                                        )
                                    ),
                                ]
                            ),
                        ]
                    ),
                    ft.Column(
                        [
                            ft.Text("Untuk memulai pengukuran.", size=14),
                            ft.Text(
                                "1. Pilih menu pengukuran lalu mulai kamera ", size=14
                            ),
                            ft.Text("2. Pertama akan terlihat kamera atas", size=14),
                            ft.Text(
                                "3. Klik stop kamera apabila bayi dalam keadaan yang baik untuk diukur",
                                size=14,
                            ),
                            ft.Text(
                                "4. Klik lanjutkan untuk melanjutkan ke kamera samping ",
                                size=14,
                            ),
                            ft.Text(
                                "5. Klik stop kamera apabila posisi bayi sudah sesuai",
                                size=14,
                            ),
                            ft.Text(
                                '6. Klik tombol "Proses pengkuran" untuk memulai proses pengkuran',
                                size=14,
                            ),
                            ft.Text(
                                "7. Hasil pengukuran akan ditampilkan secara langsung dan bisa dilihat riwayatnya pada menu riwayat",
                                size=14,
                            ),
                            ft.Text(
                                "Berikut merupakan video petunjuk pengukuran",
                            ),
                            ft.Container(
                                ft.Video(
                                    playlist=ft.VideoMedia(
                                        "images/video1.mp4"
                                    ),
                                    autoplay=False,
                                    aspect_ratio=16 / 9,
                                    expand=True,
                                    fill_color=ft.colors.INDIGO_50,
                                    filter_quality="high",
                                    on_loaded=lambda e: print("Loaded"),
                                )
                            ),
                        ]
                    ),
                    ft.Column(
                        [
                            ft.Text(
                                "Jika hasil kurang akurat ulangi langkah-langkah tadi dengan memperhatikan beberapa hal berikut.",
                                size=14,
                            ),
                            ft.Text(
                                "1. Perhatikan posisi bayi agar tidak miring dilihat dari kamera.",
                                size=14,
                            ),
                            ft.Text(
                                "2. Pastikan tangan dan kaki bayi tidak menutupi bagian tubuh yang lain ",
                                size=14,
                            ),
                            ft.Text(
                                "3. Pastikan badan bayi terlihat semuanya di dalam kamera",
                                size=14,
                            ),
                            ft.Text(
                                "4. Jika posisi bayi dirasa sudah pas, pengukuran siap dilakukan",
                                size=14,
                            ),
                        ]
                    ),
                ],
                scroll=ft.ScrollMode.AUTO,
                spacing=20,
            ),
            padding=ft.padding.all(20),
            expand=True,
        )

    def riwayat():
        lv = ft.ListView(
            spacing=20,
            padding=20,
            height=page.window_height - 150 if page.window_height else 300,
        )

        def update_view():
            # Mengambil data baru
            # result_data = load_data()
            data_history = load_json("results.json")
            if data_history:
                lv.controls.clear()  # Bersihkan kontrol lama
                for i in range(len(data_history["results"]) - 1, -1, -1):
                    lv.controls.append(
                        ft.Card(
                            ft.Container(
                                padding=ft.padding.all(20),
                                content=ft.Row(
                                    [
                                        ft.Row(
                                            [
                                                ft.Text(f"ID{i}"),
                                                ft.Text(
                                                    data_history["results"][i]["date"]
                                                ),
                                            ]
                                        ),
                                        ft.IconButton(
                                            "Download",
                                            on_click=lambda e, i=i: download_action(
                                                e, i
                                            ),
                                        ),
                                    ],
                                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                                ),
                            )
                        )
                    )
                page.update()

        # Memperbarui tampilan setiap 5 detik
        while True:
            update_view()
            time.sleep(0.2)
            return ft.Column(
                [
                    ft.Container(
                        ft.ElevatedButton(
                            "Download Excel",
                            icon=ft.icons.DOWNLOAD,
                            on_click=lambda e: download_action(e, "all"),
                        ),
                        padding=ft.padding.all(20),
                    ),
                    lv,
                ],
                horizontal_alignment=ft.CrossAxisAlignment.END,
            )

    def result():
        return ft.Container(
            padding=ft.padding.all(20),
            content=ft.Column(
                [
                    # ft.Text("Hasil Pengukuran", weight=ft.FontWeight.BOLD),
                    ft.DataTable(
                        vertical_lines=ft.BorderSide(1, ft.colors.PRIMARY),
                        columns=[
                            ft.DataColumn(
                                ft.Text("Hasil Pengukuran", weight=ft.FontWeight.BOLD)
                            ),
                            ft.DataColumn(
                                ft.Text("Nilai", weight=ft.FontWeight.BOLD),
                                numeric=True,
                            ),
                            # ft.DataColumn(ft.Text("Age"), numeric=True),
                        ],
                        rows=[
                            ft.DataRow(
                                cells=[
                                    ft.DataCell(ft.Text("Tinggi")),
                                    # ft.DataCell(ft.Text("Smith")),
                                    ft.DataCell(ft.Text(measurement_result[5])),
                                ],
                            ),
                            ft.DataRow(
                                cells=[
                                    ft.DataCell(ft.Text("Lingkar Kepala")),
                                    # ft.DataCell(ft.Text("Brown")),
                                    ft.DataCell(ft.Text(measurement_result[0])),
                                ],
                            ),
                            ft.DataRow(
                                cells=[
                                    ft.DataCell(ft.Text("Lingkar Dada")),
                                    ft.DataCell(ft.Text(measurement_result[1])),
                                ],
                            ),
                            ft.DataRow(
                                cells=[
                                    ft.DataCell(ft.Text("Lingkar Perut")),
                                    ft.DataCell(ft.Text(measurement_result[2])),
                                ],
                            ),
                            ft.DataRow(
                                cells=[
                                    ft.DataCell(ft.Text("Lingkar Kaki")),
                                    ft.DataCell(ft.Text(measurement_result[4])),
                                ],
                            ),
                            ft.DataRow(
                                cells=[
                                    ft.DataCell(ft.Text("Lingkar Lengan")),
                                    ft.DataCell(ft.Text(measurement_result[3])),
                                ],
                            ),
                        ],
                    )
                ]
            ),
        )

    def home():
# Load data dari file JSON
        data = load_json("results.json")
        now = datetime.now()
        frequency_date = defaultdict(int)
        frequency_month = defaultdict(int)

        # Ambil bulan dan tahun saat ini
        current_month = now.month
        current_year = now.year

        for result in data["results"]:
    # Ambil tanggal
            date_str = result["date"].split(",")[0]

            # Ubah format tanggal menjadi objek datetime
            try:
                date_obj = datetime.strptime(date_str, "%d %B %Y")  # sesuaikan format dengan data
            except ValueError:
                continue  # Lewati jika format tidak valid

            # Cek apakah tahun adalah tahun saat ini
            if date_obj.year == current_year:
                # Tambahkan ke frekuensi berdasarkan bulan
                month_year_key = date_obj.strftime("%B")  # Dapatkan nama bulan
                frequency_month[month_year_key] += 1
                
                # Hanya tambahkan frekuensi tanggal untuk bulan ini
                if date_obj.month == current_month:
                    formatted_date = date_obj.strftime("%d/%m/%Y")
                    frequency_date[formatted_date] += 1



        # Mengambil daftar tanggal dan bulan serta jumlahnya
        dates = list(frequency_date.keys())
        counts_date = [frequency_date[date] for date in dates]

        months = list(frequency_month.keys())
        counts_month = [frequency_month[month] for month in months]


        fig, ax = plt.subplots()

        ax.bar(dates, counts_date, color="#c7d2fe")
        ax.set_ylabel("Jumlah pengukuran")
        ax.set_xlabel("Tanggal")
        ax.set_title("Jumlah pengukuran per tanggal bulan ini")
        chart = MatplotlibChart(fig, expand=True, transparent=True)
        
        fig2, ax2 = plt.subplots()
        ax2.bar(months, counts_month, color="#c7d2fe")
        ax2.set_ylabel("Jumlah pengukuran")
        ax2.set_xlabel("Bulan")
        ax2.set_title("Jumlah pengukuran per bulan tahun ini")
        chart2 = MatplotlibChart(fig2, expand=True, transparent=True)
        return ft.Container(
            padding=ft.padding.only(top=20, left=20),
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Card(
                                content=ft.Container(
                                    content=ft.Row(
                                        [
                                            ft.Column(
                                                [
                                                    ft.Text(
                                                        "Antropometri Bayi Digital",
                                                        weight=ft.FontWeight.BOLD,
                                                    ),
                                                    ft.Text(
                                                        "Antropometri Bayi Digital menggunakan kamera yang terintegrasi dengan image processing dan deep learning",
                                                        weight=ft.FontWeight.W_100,
                                                        size=14,
                                                    ),
                                                ],
                                                width=400,
                                            ),
                                            ft.Card(
                                                content=ft.Container(
                                                    ft.Image(
                                                        src=f"images/baby.png",
                                                        width=100,
                                                        height=100,
                                                    ),
                                                    bgcolor=ft.colors.PRIMARY_CONTAINER,
                                                    border_radius=10,
                                                    padding=ft.padding.all(10),
                                                )
                                            ),
                                        ],
                                        spacing=10,
                                        tight=True,
                                        # width=600
                                    ),
                                    padding=ft.padding.all(10),
                                ),
                                # elevation=2,
                                # width=600
                            ),
                            ft.Card(
                                content=ft.Container(
                                    content=ft.Row(
                                        [
                                            ft.Image(
                                                src=f"images/tf.png",
                                                width=75,
                                            ),
                                            ft.Image(
                                                src=f"images/its.png",
                                                width=75,
                                            ),
                                        ],
                                        spacing=10,
                                        height=120,
                                        alignment=ft.MainAxisAlignment.CENTER,  # Center the Row content
                                    ),
                                    padding=ft.padding.all(10),
                                    alignment=ft.alignment.center,
                                    width=190
                                )
                            ),
                            
                        ],
                        spacing=20,
                    ),
                    ft.Container(
                        ft.Card(chart, expand=True),
                       	padding=ft.padding.only(right=20),
                        # width=1024,/
                    ),
                     ft.Container(
                        ft.Card(chart2, expand=True),
                       	padding=ft.padding.only(right=20),
                        # width=1024,
                    ),
                ],
                width=page.window.width,
                scroll=ft.ScrollMode.AUTO,
                height=page.window_height - 90 if page.window_height else 300,
                spacing=20,
            ),
        )

    def kalibrasi():
        return ft.Column(
            [
                ft.Container(expand=True),
                ft.Container(
                    ft.Row([img, img1], alignment=ft.MainAxisAlignment.CENTER),
                    # padding=ft.padding.only(left=10, right=10),
                    padding=ft.padding.all(20),
                ),
                ft.Container(
                    ft.Row(
                        [
                            ft.ElevatedButton(
                                "Mulai Kamera", on_click=start_all_cameras
                            ),
                            ft.ElevatedButton("Stop Kamera", on_click=stop_all_cameras),
                            ft.ElevatedButton(
                                "Proses Kalibrasi", on_click=capture_all_images
                            ),
                        ],
                        alignment=ft.MainAxisAlignment.CENTER,
                        spacing=20,
                    ),
                    padding=ft.padding.only(bottom=40),
                    expand=True,
                ),
            ],
            alignment=ft.MainAxisAlignment.START,
            expand=True,
        )

    def pengukuran():
        return ft.Column(
            [
                ft.Container(
                    ft.Row([img2], alignment=ft.MainAxisAlignment.CENTER),
                    # padding=ft.padding.only(left=10, right=10),
                    padding=ft.padding.all(20),
                ),
                ft.Container(
                    ft.Row(
                        [
                            ft.ElevatedButton(
                                "Mulai Kamera", on_click=start_camera_pengukuran
                            ),
                            ft.ElevatedButton(
                                "Stop Kamera", on_click=stop_camera_pengukuran1
                            ),
                            ft.ElevatedButton(
                                "Lanjutkan", on_click=capture_pengukuran1
                            ),
                        ],
                        alignment=ft.MainAxisAlignment.CENTER,
                        spacing=20,
                    ),
                    padding=ft.padding.only(bottom=20),
                ),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
        )

    def pengukuran2():
        return ft.Column(
            [
                ft.Container(
                    ft.Row([img3], alignment=ft.MainAxisAlignment.CENTER),
                    # padding=ft.padding.only(left=10, right=10),
                    padding=ft.padding.all(20),
                ),
                ft.Container(
                    ft.Row(
                        [
                            ft.ElevatedButton(
                                "Kembali", on_click=lambda e: change_content(2)
                            ),
                            ft.ElevatedButton(
                                "Stop Kamera", on_click=stop_camera_pengukuran2
                            ),
                            ft.ElevatedButton(
                                "Proses Pengukuran", on_click=capture_pengukuran2
                            ),
                        ],
                        alignment=ft.MainAxisAlignment.CENTER,
                        spacing=20,
                    ),
                    padding=ft.padding.only(bottom=20),
                ),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
        )

    def on_change(e):
        global selected_index  # Use global to modify the global variable
        selected_index = e.control.selected_index
        change_content(selected_index)
        # if(selected_index == 1 | selected_index == 2):
        if selected_index == 1:
            stop_all_cameras(None)
        if selected_index == 2:
            stop_all_cameras(None)

    img = ft.Image(
        src="./frames/frame0.png",
        # width=400,
        height=250,
        fit=ft.ImageFit.CONTAIN,
        gapless_playback=True,
    )

    img1 = ft.Image(
        src="./frames/frame0.png",
        # width=00,
        height=250,
        fit=ft.ImageFit.CONTAIN,
        gapless_playback=True,
    )
    img2 = ft.Image(
        src="./frames/frame0.png",
        # width=00,
        height=400,
        fit=ft.ImageFit.CONTAIN,
        gapless_playback=True,
    )
    img3 = ft.Image(
        src="./frames/frame0.png",
        # width=00,
        height=400,
        fit=ft.ImageFit.CONTAIN,
        gapless_playback=True,
    )
    rail = ft.NavigationRail(
        selected_index=0,
        label_type=ft.NavigationRailLabelType.ALL,
        min_width=100,
        min_extended_width=200,
        group_alignment=-0.9,
        destinations=[
            ft.NavigationRailDestination(
                label="Home",
                icon=ft.icons.HOME_OUTLINED,
                selected_icon_content=ft.Icon(ft.icons.HOME),
            ),
            ft.NavigationRailDestination(
                icon_content=ft.Icon(ft.icons.LINEAR_SCALE_OUTLINED),
                label="Kalibrasi",
                selected_icon=ft.icons.LINEAR_SCALE_ROUNDED,
            ),
            ft.NavigationRailDestination(
                icon_content=ft.Icon(ft.icons.CAMERA_OUTLINED),
                label="Pengukuran",
                selected_icon=ft.icons.CAMERA,
            ),
            ft.NavigationRailDestination(
                icon_content=ft.Icon(ft.icons.HISTORY_OUTLINED),
                label="Riwayat",
                selected_icon=ft.icons.MANAGE_HISTORY,
            ),
            ft.NavigationRailDestination(
                icon_content=ft.Icon(ft.icons.BOOK_OUTLINED),
                label="Petunjuk",
                selected_icon=ft.icons.BOOK,
            ),
        ],
        on_change=on_change,
    )

    content_container = ft.Column(expand=True)
    page.add(
        ft.Row([rail, ft.VerticalDivider(width=1), content_container], expand=True)
    )

    page.dark_theme = ft.Theme(color_scheme_seed="indigo")
    page.theme = ft.Theme(color_scheme_seed="indigo")
    page.theme_mode = ft.ThemeMode.LIGHT
    page.update()

    change_content(0)


ft.app(main, assets_dir="assets")

