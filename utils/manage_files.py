import gzip
import re
import shutil
import os
import os.path
import typing
import urllib.request
import xml.etree.ElementTree as ET

import pandas as pd
from tqdm import tqdm


def download_files(url: str) -> str:
    # Extract from the url the file name to download
    prog = re.compile(r"\/files\/(?P<name>.*?)\?")
    re_search = prog.search(url)
    assert re_search is not None

    # The file name is in the capture group "name"
    download_name = re_search.group("name")
    # split the file name in its name without the extension and its extension
    base_name, download_ext = download_name.rsplit(".", maxsplit=1)

    # if the extension is an archive then the file needs to be split
    download = None
    if download_ext in ["gz", "zip"]:
        zip_name = download_name
        # the file name unzipped is the base name
        download = DownloadFile(base_name, zip_name, url)
    else:
        # the file name is not an archive so the downloaded file has the correct name
        download = DownloadFile(download_name, url=url)

    print(
        f"\nDownloading {'and extracting ' if download.zip_name is not None else ''}files"
    )
    download()
    return download.file_name


def open_df(file_name: str, names=None, sep=None) -> pd.DataFrame:
    ext = file_name.rsplit(".", maxsplit=1)[1]

    df = None
    if ext == "jsonl":
        df = pd.read_json(file_name, lines=True)
    else:
        assert names is not None
        assert sep is not None
        df = pd.read_csv(file_name, names=names, sep=sep)

    assert isinstance(df, pd.DataFrame)
    return df


def open_xml(file_name: str) -> typing.List[str]:
    mytree = ET.parse(file_name)
    myroot = mytree.getroot()

    topics = list()
    for item in myroot:
        for x in item:
            if x.tag == "title":
                assert isinstance(x.text, str)
                topics.append(x.text.strip())
    return topics


class DownloadFile:
    base_dir: str
    url: str
    file_name: str
    zip_name: str

    def __init__(
        self,
        file_name: str,
        zip_name: str = None,
        url: str = None,
        base_dir: str = None,
    ) -> None:
        default_dir = os.path.join(os.getcwd(), "downloads")
        self.base_dir = default_dir if base_dir is None else base_dir

        self.file_name = self.validate_name(file_name)
        self.zip_name = self.validate_name(zip_name, "zips")
        self.url = url

    def validate_name(self, file_name: str, prefixes: str = ""):
        """The file name (either zip or folder/file name) must
        be without any path prefixes (e.g. parent folders). The parent
        directory is then added as a prefix of the file name, in this way
        paths are similar to each other, i.e. having the same prefix."""
        if file_name is None:
            return None

        assert len(file_name.split("/")) == 1
        base_folder = os.path.join(self.base_dir, prefixes)
        os.makedirs(base_folder, exist_ok=True)

        return os.path.join(base_folder, file_name)

    def tqdm_copy_file_obj(self, src, dest, length: int):
        with tqdm(
            total=length,
            unit="B",
            unit_scale=True,
            desc="Downloading file",
        ) as pbar:
            chunk_size = 1024
            while True:
                chunk = src.read(chunk_size)
                if not chunk:
                    break
                dest.write(chunk)
                pbar.update(len(chunk))

    def download(self):
        """Download the file from the url."""
        assert self.url is not None
        file_path = self.file_name if self.zip_name is None else self.zip_name

        if not os.path.isfile(file_path):
            with urllib.request.urlopen(self.url) as src, open(file_path, "wb") as dest:
                self.tqdm_copy_file_obj(src, dest, int(src.info()["Content-Length"]))
        else:
            print(f"File '{file_path}' already present")

    def __unzip(self):
        """Unzip the file."""
        assert self.zip_name is not None

        try:
            shutil.unpack_archive(self.zip_name, extract_dir=self.file_name)
            print(f"'{self.zip_name}' unzipped in '{self.file_name}'")
        except ValueError as e:
            print(f"File {self.zip_name} invalid")

    def __ungzip(self):
        """Extract a gz archive"""
        assert self.zip_name is not None

        with gzip.open(self.zip_name, "rb") as src_file:
            with open(self.file_name, "wb") as dest_file:
                dest_file.write(src_file.read())
        print(f"'{self.zip_name}' unzipped in '{self.file_name}'")

    def uncompress(self):
        assert self.zip_name is not None
        _, ext = self.zip_name.rsplit(".", maxsplit=1)
        if ext == "gz":
            self.__ungzip()
        elif ext == "zip":
            self.__unzip()
        else:
            print("Format not supported")

    def __call__(self):
        """Download and unzip only if the respective fields are set (e.g. url or zip name)"""
        if not os.path.exists(self.file_name):
            if self.url is not None:
                self.download()
            if self.zip_name is not None:
                self.uncompress()
        else:
            print(f"'{self.file_name}' already present")
