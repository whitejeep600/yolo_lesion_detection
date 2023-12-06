# Lesion detection in medical images with Deep Learning computer vision models
## Final project in Medical Image Processing, NTU, 2023
### Team: Antoni Maciąg (馬安德), R11922182

Automatic detection of abnormalities is a common use-case across different domains of Artificial
Intelligence. In Medical Image Processing, a broad class of such abnormalities is described
as lesions - tissue fragments that have suffered various kinds of damage, such as a wound,
fracture or even a tumor. Lesion detection has clear real-world applications in improving the
efficiency and accuracy of diagnosing, drawing medical professionals' attention to tissue
fragments that require careful examination, saving time looking through images that do not
represent any tissue damage, and providing an additional check to the professional's expertise.
It is also worth noting that all of these benefits refer to a discipline in which a mistake,
such as misclassifying a tumor as healthy tissue, can have very serious consequences, and in
which the professionals' time is costly and - in many regions - in short supply.

Motivated by the above, I have decided to check the efficacy of applying Deep Learning methods
(as of 2023, the undisputed state-of-the-art solution in virtually all detection and
classification tasks) to automatic detection of lesions in medical images.

## Dataset

The most common bottleneck in Deep Learning projects is the access to datasets of sufficient
size and quality, which are difficult to obtain, as they usually require costly manual labeling.
In Medical Image Processing, this problem is compounded by the fact that labeling has to be done
by highly skilled professionals. Fortunately, I was able to find and use the 
[DeepLesion](https://nihcc.app.box.com/v/DeepLesion) dataset with 32,735 lesions, mostly with
one lesion per image. These lesions are additionally annotated with the location or kind of
tissue they appear in. The types are: abdomen, mediastinum, liver, lung,
kidney, soft tissue, bone, and pelvis (apparently not a bone). The labeling, however, has two 
drawbacks:

- labels correspond to the type of tissue, and not to the kind of lesion itself (like tumor, 
mechanical damage, wound, fracture). I imagine
a medical professional would be more interested in determining the kind of the lesion, and
thus its seriousness, potential treatment etc. They hopefully know what they are photographing
anyway, so the tissue of the lesion is not a useful bit of additional information.
- the annotations can only be found for the evaluation and test splits, rather than the train
split. Because of this, it is impossible to train a detector that would even provide the
information about the lesion's tissue. This is all the more surprising as, I imagine,
in contrast to costly lesion type annotations, the information about tissue type should be easily
accessible based on what part of the patient a given image was obtained from (which is known
at the time of photographing).

One thing we can still do with annotations as described is to train a non-classifying lesion
detector, and then on the test set, compare its performance on different kinds of tissues. This
way we can determine whether lesions in certain tissue types are easier or harder to classify
than others. For good DL models, this should also align with human experience, and guide
the professionals to pay particular attention to certain tissues.

## Model

With choosing the specific Deep Learning model to apply, we have much more freedom than with
choosing the dataset, as there is a plethora of carefully designed, pretrained and freely available
models. I choose the YOLOv5 model by Ultralytics - a model based on a deep Convolutional
Neural Network, with a good reputation for training and inference speed, as well as high
accuracy. It is available in sizes ranging from Small, through Medium and Large, to XLarge. As
a balance between performance and speed, I chose the Medium model and decided to fine-tune
the pretrained weights released by Ultralytics.

## Training

I used the default hyperparameters for training YOLOv5 on the Objects365 dataset. The training
was run for 100 epochs, which took about 10 hours on two NVIDIA GeForce GTX 1080 Ti GPUs with a
batch size of 64.

Note that the training was not run directly on images from the DeepLesion dataset. Preprocessing
to the format expected by YOLO had to be done, and in addition, I had to increase the contrast
of the images (this problem is also mentioned on the dataset's website). For more detailed
descriptions of these steps, please refer to `README.md` (or the code, after all it describes best
what it's doing).

## Results

First of all it has to be said that the results turned out not to be sufficiently good
for any kind of medical application. Below I will try to analyze what the reasons could be,
and draw some conclusions.

The recall on the test set was 0.43, and the false positive rate was 0.54. Dividing by tissue
type (again, this refers only to the labels, as the model's detections do not distinguish
the types), the recall scores were, as follows (sorted):

- lung: 0.69,
- mediastinum: 0.59, 
- bone: 0.50,
- soft tissue: 0.43,
- pelvis: 0.42
- abdomen: 0.30,
- kidney: 0.29, 
- liver: 0.07.

We can see the range that most scores fall into is 0.3 to 0.5, with positive exceptions being
the lungs and mediastinum (physically close to each other, accidentally or not), and the negative
exception being the liver, with much worse performance than the others.

We may now consider why even the best results are far below what would be applicable in a
real-world system. I can identify the following possible reasons:

- label quality. The images come in groups made during the same examination, representing
subsequent slices of the patient in small distances on the axis perpendicular to the image plane.
While I am obviously not a medical professional, to my eye it looks like the regions
corresponding to the same 3D lesion in different slices are labeled differently in some of the
slices. I have included an example of this as the `images/inconsistent{1,2}.png` images. These
are images from the dataset with overlaid labels (blue) and model predictions (red in this case,
green if they were correct). The region labeled as a lesion in `inconsistent2.png` also appears
(partly) in the other picture, where we can actually see the model's prediction. However,
the lesion is no longer labeled as such in that picture. If I am correct, such noisy labels can
interfere with both the training and the evaluation of the model.
- problem hardness, simply. Looking at the images from the dataset, it is not immediately clear
to the human (if unqualified) eye where the lesion is and even with the annotation, it is
not always obvious how the professional spotted the lesion and what makes it different
from other, similar regions in the image. It should not be surprising, then, that a DL model
fails to achieve satisfactory performance, as these rarely outperform humans. This also means 
I might be completely wrong about the previous point -  which, however, would be all the more
evidence for the hardness of the problem. 
- contrast processing. As mentioned, the original images from the dataset had very low contrast,
with pixel grayness values only occupying a narrow range. This is supposed to be expected, and the
reasons and preprocessing instructions are described on the DeepLesion website. However, I do
not think these instructions make sense, or in any case I did not understand them, so I had to
come up with a simple method of increasing the contrast myself. While the resulting images
look reasonable, they still only have a small set of possible grayness values (this set is
stretched between 0 and 255, so as wide as possible. What I mean by "small" here is that
only a few different values from that range are assumed). It is possible the performance
would improve with images closer to the original ones. Further evidence for this is that some
lesions are not visible to at all - vide `images/sneaky_lesion.png`. I guess the annotating
professional must have had access to a higher-contrast image to make the annotation.
- model pretraining. The YOLO model was not pretrained to deal with gray medical images
representing 2D slices of the human body, but to deal with images of 3D scenes from the real
world (I mean the world as it appears to human eyes). Therefore, this project represents a bit
of a domain adaptation problem, in which case a performance drop in comparison
to the original domain is absolutely expected.

To conclude the report on a bit of a more optimistic note, under `images/correct{1,2}` we can
see two cases in which the model's predictions were actually correct. Still, it is not immediately
clear what made the model arrive at them, and how the regions detected as lesions are different
from other parts of the images.