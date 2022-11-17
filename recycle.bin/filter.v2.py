from struct import unpack
import os

# NOTE : https://stackoverflow.com/questions/62586443/tensorflow-error-when-trying-transfer-learning-invalid-jpeg-data-or-crop-windo

marker_mapping = {
    0xFFD8: "Start of Image",
    0xFFE0: "Application Default Header",
    0xFFDB: "Quantization Table",
    0xFFC0: "Start of Frame",
    0xFFC4: "Define Huffman Table",
    0xFFDA: "Start of Scan",
    0xFFD9: "End of Image",
}


class JPEG:
    def __init__(self, image_file):
        with open(image_file, "rb") as f:
            self.img_data = f.read()

    def decode(self):
        data = self.img_data
        while True:
            (marker,) = unpack(">H", data[0:2])
            # print(marker_mapping.get(marker))
            if marker == 0xFFD8:
                data = data[2:]
            elif marker == 0xFFD9:
                return
            elif marker == 0xFFDA:
                data = data[-2:]
            else:
                (lenchunk,) = unpack(">H", data[2:4])
                data = data[2 + lenchunk :]
            if len(data) == 0:
                break


bad_paths = []

images = []  # list

for folder_name in ("Cats", "Dogs"):
    folder_path = os.path.join("downloads", "CatsDogs", folder_name)
    for fname in os.listdir(folder_path):
        image = os.path.join(folder_path, fname)

        ij = JPEG(image)
        try:
            ij.decode()
        except:
            print(f"Found bad path {image}")
            bad_paths.append(image)

print(len(bad_paths))

for pth in bad_paths:
    try:
        os.remove(pth)
        print(pth, "removed")
    except Exception as e:
        print("FATAL ERROR @", pth, ":", e)
