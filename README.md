# Dubai Satellite Imagery Semantic Segmentation Using Deep Learning

Fork [README](https://github.com/prodramp/DeepWorks/blob/main/README.md)

## Dataset
[Humans in the Loop](https://humansintheloop.org/) has published an open access dataset annotated for a joint project with the [Mohammed Bin Rashid Space Center](https://www.mbrsc.ae/) in Dubai, the UAE.
The dataset consists of aerial imagery of Dubai obtained by MBRSC satellites and annotated with pixel-wise semantic segmentation in 6 classes.
The images were segmented by the trainees of the Roia Foundation in Syria.


### Semantic Annotation

The images are densely labeled and contain the following 6 classes:

| Name       | R   | G   | B   | Color                                                                                              |
| ---------- | --- | --- | --- | -------------------------------------------------------------------------------------------------- |
| Building   | 60  | 16  | 152 | <p align="center"><img width = "30" height= "20" src="./readme_images/label_building.png" /></p>   |
| Land       | 132 | 41  | 246 | <p align="center"><img width = "30" height= "20" src="./readme_images/label_land.png" /></p>       |
| Road       | 110 | 193 | 228 | <p align="center"><img width = "30" height= "20" src="./readme_images/label_road.png" /></p>       |
| Vegetation | 254 | 221 | 58  | <p align="center"><img width = "30" height= "20" src="./readme_images/label_vegetation.png" /></p> |
| Water      | 226 | 169 | 41  | <p align="center"><img width = "30" height= "20" src="./readme_images/label_water.png" /></p>      |
| Unlabeled  | 155 | 155 | 155 | <p align="center"><img width = "30" height= "20" src="./readme_images/label_unlabeled.png" /></p>  |


### Sample Images & Masks

<p align="center"><img width = "95%" height= "auto" src="./readme_images/sample_image_t8_004.jpg" /></p>

<p align="center"><img width = "95%" height= "auto" src="./readme_images/sample_image_t8_003.jpg" /></p>

<p align="center"><img width = "95%" height= "auto" src="./readme_images/sample_image_t4_001.jpg" /></p>

<p align="center"><img width = "95%" height= "auto" src="./readme_images/sample_image_t6_002.jpg" /></p>
