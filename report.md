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
classification tasks) to automatically detecting lesions in medical images.

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
- mechanical damage, wound, fracture). I imagine
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


introducing yolo
maybe some data processing, training procedure
metrics
false positive rate 0.54, general recall 0.43,
recall per class: {'bone': 0.5, 'abdomen': 0.30, 'mediastinum': 0.59, 
'liver': 0.07, 'lung': 0.69, 'kidney': 0.29, 
'soft': 0.43, 'pelvis': 0.42}
findings, conclusions, representative images  (to an images/ folder but actually
include them here in the report)