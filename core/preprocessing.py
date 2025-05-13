import pandas as pd
from xml.etree import ElementTree as ET


def get_processed_annotations(images_set: str, xml_path: str) -> pd.DataFrame:
    """
    Creates a pandas dataframe for given image set and matchable markup
    :param images_set: path to the txt-file with image indexes
    :param xml_path: path to the current directory that contains xml-markup
    """
    with open(file=images_set, mode="r") as file:
        f = file.read().split("\n")

        return pd.concat(
            [
                parse_xml_anns(path=f"{xml_path}/{img_id}.xml", img_id=img_id)
                for img_id in f
            ],
            ignore_index=True,
        )


def parse_xml_anns(path: str, img_id: str) -> pd.DataFrame:
    """
    Function that extracts and transforms annotations from
    given XML files with detection markup (bboxes)
    :param path: path to the current directory that contains xml-markup
    :param img_id: sample index to match markup file
    """

    anns = {
        "image_id": [],
        "boxes": [],
        "labels": [],
        "xmin": [],
        "ymin": [],
        "xmax": [],
        "ymax": [],
    }

    tree = ET.parse(path)
    root = tree.getroot()

    objects = root.findall("object")

    for o in objects:
        label = o.find("name").text
        bbox = o.find("bndbox")

        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        anns["image_id"].append(int(img_id))
        anns["labels"].append(label)
        anns["boxes"].append([xmin, ymin, xmax, ymax])

        anns["xmin"].append(xmin)
        anns["ymin"].append(ymin)
        anns["xmax"].append(xmax)
        anns["ymax"].append(ymax)

    return pd.DataFrame.from_dict(anns)
