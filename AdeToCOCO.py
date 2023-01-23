# ref: https://github.com/facebookresearch/Mask2Former/blob/main/datasets/prepare_ade20k_ins_seg.py

import json, os, pickle
import pycocotools.mask as mask_util
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

def pickleload(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def save_json(d, file):
    with open(file, 'w') as f:
        json.dump(d, f)

class AdeToCOCO():
    def __init__(self, pklPath, datasetDir, objectNames) -> None:
        self.statics = pickleload(pklPath)
        self.datasetDir = datasetDir
        self.objectNames = objectNames
        self.annId = 1
    
    def getObjectId(self,name):
        objId = np.where(np.array(self.statics["objectnames"]) == name)[0][0]
        # print(f"id of {name} is {objid}")
        return int(objId)

    def getImageIds(self,names):
        imgIds = np.array([], dtype=int)
        self.statics["filename"]
        for name in names:
            objId = self.getObjectId(name)
            id = np.where(np.array(self.statics["objectPresence"][objId]) > 0)[0]
            imgIds = np.hstack((imgIds, id))
        return np.unique(imgIds).tolist()

    def getImagePath(self,imageId):
        path = Path(self.datasetDir) / self.statics["folder"][imageId] / self.statics["filename"][imageId]
        assert path.exists(), f"Image file not exist"
        return str(path)

    def getInfoJson(self,imageId):
        path = Path(self.datasetDir) / self.statics["folder"][imageId] / self.statics["filename"][imageId].replace("jpg","json")
        assert path.exists(), f"Image information json file not exist"
        return str(path)

    # return coco annotation format
    def generateAnnotations(self,imageId, imageInfo):
        objects = imageInfo["object"]

        annotations = []

        for obj in objects:
            
            if obj["name"] not in objectNames:
                continue

            annotation = {
            "id": int(self.annId),
            "image_id": int(imageId),
            "category_id": int(obj["name_ndx"]),
            "segmentation": [],
            "area": float,
            "bbox": [],
            "iscrowd": int(0)
            }

            # trans polygan
            polygon = obj["polygon"]
            xmin, xmax = 1e8, -1e8
            ymin, ymax = 1e8, -1e8
            for x,y in zip(polygon['x'], polygon['y']):
                annotation["segmentation"].extend([x,y])
                xmin, xmax = min(xmin, x), max(xmax, x)
                ymin, ymax = min(ymin, y), max(ymax, y)

            # # get mask 
            # maskPath = Path(datasetdir) / imageInfo["folder"] / obj["instance_mask"]
            # mask = np.asarray(Image.open(maskPath))
            # assert mask.dtype == np.uint8
            # inds = np.nonzero(mask)
            # ymin, ymax = inds[0].min(), inds[0].max()
            # xmin, xmax = inds[1].min(), inds[1].max()

            # calculate bbox

            annotation["bbox"] = [int(xmin), int(ymin), int(xmax - xmin + 1), int(ymax - ymin + 1)]
            # get rle
            # rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            h, w = imageInfo["imsize"][0], imageInfo["imsize"][1]
            poly = [annotation["segmentation"]]
            rle = mask_util.frPyObjects(poly, h,w)[0]

            rle["counts"] = rle["counts"].decode("utf-8")
            annotation["segmentation"] = rle
            
            # getarea
            annotation["area"] = int(mask_util.area(rle))

            # print(annotation)
            # raise
            annotations.append(annotation)

            
            self.annId += 1
        return annotations

    # return coco image format
    def generateImage(self,imageId, imagePath, imageInfo):
        image = {
        "id":int,
        "file_name":str,
        "width":int,
        "height":int
        }
        image["id"] = int(imageId)
        image["file_name"] = imagePath
        image["width"] = int(imageInfo["imsize"][1])
        image["height"] = int(imageInfo["imsize"][0])
        return image


    def convert(self):
        # Convert Category
        adeCategories = []
        for name in self.objectNames:
            category_dict  = {"id":int,"name":str}
            id = self.getObjectId(name) + 1 # consist with seg json name_ndx
            category_dict["id"] = id
            category_dict["name"] = name
            adeCategories.append(category_dict)

        # print(adeCategories)

        train_dict = {}
        val_dict = {}
        
        train_images= []
        train_category = adeCategories
        train_annotations =[]

        val_images = []
        val_category = adeCategories
        val_annotations = []
        decode_fail = 0

        for imgId in tqdm(self.getImageIds(objectNames)):
            jsonFile = self.getInfoJson(imgId)

            # TODO: handle decode fail
            with  open(jsonFile, 'r', encoding='utf-8') as f:
                try:
                    imageInfo = json.load(f)['annotation']
                except:
                    print(f"fail to decode {jsonFile}")
                    decode_fail += 1
                    continue
            
            imagePath = self.getImagePath(imgId)
            # print(imagePath)
            image = self.generateImage(imgId, imagePath, imageInfo)
            annotations = self.generateAnnotations(imgId, imageInfo)
            if "ADE/training" in imagePath:
                train_images.append(image)
                train_annotations.extend(annotations)
            elif "ADE/validation" in imagePath:
                val_images.append(image)
                val_annotations.extend(annotations)
            # else:
            #     print(imagePath)

        train_dict["images"] = train_images
        train_dict["categories"] = train_category
        train_dict["annotations"] = train_annotations

        val_dict["images"] = val_images
        val_dict["categories"] = val_category
        val_dict["annotations"] = val_annotations

        # print(train_annotations)
        train_out_file = Path(self.datasetDir) / f"ADE20K_2021_17_01/ade20k_instance_train.json"
        val_file = Path(self.datasetDir) / f"ADE20K_2021_17_01/ade20k_instance_val.json"
        save_json(train_dict, train_out_file)
        save_json(val_dict, val_file)

if __name__ == "__main__":
    # datasetDir = os.getenv("DETECTRON2_DATASETS", "datasets")
    datasetDir = "/project/n/p10922001/DoorSeg/datasets"
    objectNames = ["door", "door frame"]
    # objectNames = ["wall"]
    # output_folder = "/project/n/p10922001/DoorSeg/datasets"
    pklPath = "/project/n/p10922001/DoorSeg/datasets/ADE20K_2021_17_01/index_ade20k.pkl"
    converter = AdeToCOCO(pklPath, datasetDir, objectNames)
    print("Start Converting.....")
    converter.convert()