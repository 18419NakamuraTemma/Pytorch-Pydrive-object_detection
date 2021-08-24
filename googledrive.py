import os
import time
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import detect
import glob
import shutil

gauth = GoogleAuth()
gauth.LocalWebserverAuth()


drive = GoogleDrive(gauth)
imgs_files = glob.glob("imgs/*")
output_files = glob.glob("det/*")

drive_tempfolder_id = 'drive folder id' # 画像の場所
drive_savefolder_id = 'drive folder id' # 画像の保存場所（ドライブ）
drive_outputfolder_id = 'drive folder id' # 物体検知で出力されたのファイル保存場所
save_folder = os.getcwd() + "/imgs" # 画像の保存場所（ローカル）


def change_folder(file_id,folder_id):
    f = drive.CreateFile({'id': file_id})
    f['parents'] = [{'id': folder_id}]
    f.Upload()

def upload_file(file_path,folder_id):
    f = drive.CreateFile({"parents": [{"id": folder_id}]})
    f.SetContentFile(file_path)
    f['title'] = os.path.basename(file_path)
    f.Upload()


def download_recursively(save_folder, drive_folder_id):
    # 保存先フォルダがなければ作成
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    max_results = 100
    query = "'{}' in parents and trashed=false".format(drive_folder_id)

    for file_list in drive.ListFile({'q': query, 'maxResults': max_results}):
        for file in file_list:
            # mimeTypeでフォルダか判別
            if file['mimeType'] == 'application/vnd.google-apps.folder':
                download_recursively(os.path.join(save_folder, file['title']), file['id'])
            else:
                file.GetContentFile(os.path.join(save_folder, file['title']))

while True:

    download_recursively(save_folder, drive_tempfolder_id)

    detect.main()

    for f in drive.ListFile({'q': '"{}" in parents'.format(drive_tempfolder_id)}).GetList():
        change_folder(f['id'],drive_savefolder_id)

    for file in imgs_files:
        upload_file(file,drive_savefolder_id)
    for file in output_files:
        upload_file(file,drive_outputfolder_id)

    target_dir = 'imgs'
    shutil.rmtree(target_dir)
    os.mkdir(target_dir)

    target_dir = 'det'
    shutil.rmtree(target_dir)
    os.mkdir(target_dir)
    time.sleep(30)



