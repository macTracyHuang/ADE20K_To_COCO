# Reference: https://github.com/facebookresearch/Mask2Former/blob/main/datasets/prepare_ade20k_ins_seg.py
# ADE20K dataset: https://groups.csail.mit.edu/vision/datasets/ADE20K/
from pathlib import Path
import argparse
import json
import pickle
import numpy as np
from tqdm import tqdm
import pycocotools.mask as mask_util

# For demo
from detectron2.data import MetadataCatalog, DatasetCatalog
import cv2, random
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt
from detectron2.data.datasets import register_coco_instances


def pickleload(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def saveJson(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)


class AdeToCOCO():
    """ A class to convert ADE20K to COCO format
    Attributes:
        statics (dict): ADE20K index pickle file data
        datasetDir (str): path to the ADE20K dataset directory
        objectNames (list): list of object names to convert
        annId (int): annotation id start from 1
    -----------------
    ADE20K index pickle data structure:
    N: number of images, C: number of object categories
    statics["filename"]: list of image file name with size N
    statics["folder"] : list of image folder name with size N
    statics["objectnames"]: list of object names with size C
    statics["objectPresence"]: list of object presence with size CxN, 
                                objectPresence(c,i) = n means image i contains n objects of category c
    """

    def __init__(self, pklPath, datasetDir, objectNames):
        """
        Args:
            pklPath (str): path to the ADE20K index pickle file
            datasetDir (str): path to the ADE20K dataset directory
            objectNames (list): list of object names to convert
        """
        self.statics = pickleload(pklPath)
        self.datasetDir = datasetDir
        self.objectNames = objectNames
        self.annId = 1

    def getObjectIdbyName(self, name):
        """Get object id by object name
        
        Args:
            name (str): object name
        Returns:
            objId (int): object id
        """
        objId = np.where(np.array(self.statics["objectnames"]) == name)[0][0]
        # print(f"id of {name} is {objid}")
        return int(objId)

    def getImageIds(self, names):
        """Get image ids by object names
        
        Args:
            names (list): list of object names
        Returns:
            imgIds (list): list of image ids
        """
        all_image_ids = []

        for name in names:
            objId = self.getObjectIdbyName(name)
            current_image_ids = np.where(
                self.statics["objectPresence"][objId] > 0)[0]
            all_image_ids.append(current_image_ids)

        imgIds = np.unique(np.concatenate(all_image_ids))
        return imgIds.tolist()

    def getImagePathbyId(self, imageId):
        """Get image path by image id
        
        Args:
            imageId (int): image id
        Returns:
            path (str): image path
        """
        path = Path(self.datasetDir) / \
            self.statics["folder"][imageId] / self.statics["filename"][imageId]
        assert path.exists(), f"Image file not exist"
        return str(path)

    def getInfoJsonbyId(self, imageId):
        """Get image information json file path by image id,
        Each image has a json file to store image information
        
        Args:
            imageId (int): image id
        Returns:
            path (str): image information json file path
        """
        path = Path(self.datasetDir) / self.statics["folder"][imageId] / \
            self.statics["filename"][imageId].replace("jpg", "json")
        assert path.exists(), f"Image information json file not exist"
        return str(path)

    def generateAnnotations(self, imageId, imageInfo):
        """ Generate annotations for a single image in COCO format
        Args:
            imageId (int): image id
            imageInfo (dict): image information
        Returns:
            annotations (list): list of annotations
        """
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

            # trans polygan to segmentation
            polygon = obj["polygon"]
            xmin, xmax = 1e8, -1e8
            ymin, ymax = 1e8, -1e8
            for x, y in zip(polygon['x'], polygon['y']):
                annotation["segmentation"].extend([x, y])
                xmin, xmax = min(xmin, x), max(xmax, x)
                ymin, ymax = min(ymin, y), max(ymax, y)

            # calculate bounding box
            annotation["bbox"] = [
                int(xmin),
                int(ymin),
                int(xmax - xmin + 1),
                int(ymax - ymin + 1)
            ]
            # get rle (Run-Length Encoding)
            # rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            h, w = imageInfo["imsize"][0], imageInfo["imsize"][1]
            poly = [annotation["segmentation"]]
            rle = mask_util.frPyObjects(poly, h, w)[0]

            rle["counts"] = rle["counts"].decode("utf-8")
            annotation["segmentation"] = rle

            # get area
            annotation["area"] = int(mask_util.area(rle))

            # print(annotation)
            annotations.append(annotation)
            self.annId += 1
        return annotations

    def generateImage(self, imageId, imagePath, imageInfo):
        """ Generate image information for a single image in COCO format
        Args:
            imageId (int): image id
            imagePath (str): image path
            imageInfo (dict): image information in ADE20K format
        Returns:
            image (dict): image information in COCO format
        """
        image = {"id": int, "file_name": str, "width": int, "height": int}
        image["id"] = int(imageId)
        image["file_name"] = imagePath
        image["width"] = int(imageInfo["imsize"][1])
        image["height"] = int(imageInfo["imsize"][0])
        return image

    def convert(self):
        # Convert Category
        adeCategories = []
        for name in self.objectNames:
            print(f"Convert {name}")
            categoryDict = {"id": int, "name": str}
            id = self.getObjectIdbyName(
                name) + 1  # consist with seg json name_ndx
            categoryDict["id"] = id
            categoryDict["name"] = name
            adeCategories.append(categoryDict)

        trainDict = {}
        valDict = {}

        trainImages = []
        trainCategory = adeCategories
        trainAnnotations = []

        valImages = []
        valCategory = adeCategories
        valAnnotations = []
        decodeFailCount = 0

        for imgId in tqdm(self.getImageIds(objectNames)):
            jsonFile = self.getInfoJsonbyId(imgId)

            # TODO: handle decode fail
            with open(jsonFile, 'r', encoding='utf-8') as f:
                try:
                    imageInfo = json.load(f)['annotation']
                except:
                    print(f"fail to decode {jsonFile}")
                    decodeFailCount += 1
                    continue

            imagePath = self.getImagePathbyId(imgId)
            # print(imagePath)
            image = self.generateImage(imgId, imagePath, imageInfo)
            annotations = self.generateAnnotations(imgId, imageInfo)
            if "ADE/training" in imagePath:
                trainImages.append(image)
                trainAnnotations.extend(annotations)
            elif "ADE/validation" in imagePath:
                valImages.append(image)
                valAnnotations.extend(annotations)
            else:
                print(f"{imagePath} is not in training or validation set")

        trainDict["images"] = trainImages
        trainDict["categories"] = trainCategory
        trainDict["annotations"] = trainAnnotations

        valDict["images"] = valImages
        valDict["categories"] = valCategory
        valDict["annotations"] = valAnnotations

        # print(trainAnnotations)
        trainOutputFilePath = Path(self.datasetDir) / \
            f"ADE20K_2021_17_01/ade20k_instance_train.json"
        valOutputFilePath = Path(self.datasetDir) / \
            f"ADE20K_2021_17_01/ade20k_instance_val.json"
        saveJson(trainDict, trainOutputFilePath)
        saveJson(valDict, valOutputFilePath)


class DemoTest():

    def __init__(self, datasetDir):
        """ A class to run demo to check the converted COCO format
        Args:
            datasetDir (str): path to the ADE20K dataset directory
        """
        self.datasetDir = datasetDir

    def startDemo(self):
        datasetName = "ade20k2021_train"
        trainJsonFilePath = Path(datasetDir) / \
            f"ADE20K_2021_17_01/ade20k_instance_train.json"
        register_coco_instances(datasetName, {}, trainJsonFilePath, datasetDir)
        dataset = DatasetCatalog.get(datasetName)
        for data in random.sample(dataset, 3):
            fileName = data["file_name"]
            img = cv2.imread(fileName)
            visualizer = Visualizer(img[:, :, ::-1],
                                    metadata=MetadataCatalog.get(datasetName))
            out = visualizer.draw_dataset_dict(data)
            plt.title(fileName.split('/')[-1])
            plt.imshow(
                cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert ADE20K to COCO format")
    parser.add_argument("--datasetDir",
                        type=str,
                        required=True,
                        help="Path to the ADE20K dataset directory")
    parser.add_argument(
        "--pklPath",
        type=str,
        required=True,
        help="Path to the ADE20K index pickle file (index_ade20k.pkl)")
    parser.add_argument("--objectNames",
                        type=str,
                        nargs='+',
                        required=True,
                        help="List of object names to convert")
    parser.add_argument("--demo",
                        type=bool,
                        default=False,
                        help="Run demo after converting")

    args = parser.parse_args()

    datasetDir = args.datasetDir
    objectNames = args.objectNames
    pklPath = args.pklPath
    print(f"Convert {objectNames} in {datasetDir}")
    converter = AdeToCOCO(pklPath, datasetDir, objectNames)
    print("Start Converting.....")
    converter.convert()
    print("Finish Conversion")

    if args.demo:
        print("Start Demo.....")
        test = DemoTest(datasetDir)
        test.startDemo()
        print("Finish Demo")
