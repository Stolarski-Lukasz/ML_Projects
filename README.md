# Example scripts used in my machine learning projects

This repository contains scripts that were used in some of my machine learning projects. The scripts were written using object-oriented programming (OOP) and adhere to SOLID principles. The file named "ml_module.py" contains classes that are imported and used in the main executable file. Currently, the following examples of the executable file are provided:

1. "run_dnn_regression.py" - an example of a DNN regression model building pipeline. It was used for VowelMeter, a program for semi-automatic placement of vowels in the IPA Vowel Diagram .You can visit the program at [vowelmeter.pythonanywhere.com](https://vowelmeter.pythonanywhere.com) and see the source code at [github.com/Stolarski-Lukasz/VowelMeter](https://github.com/Stolarski-Lukasz/VowelMeter). The model was trained on articulations of Cardinal Vowels recorded by 14 phoneticians, which were normalized in Miller’s (1989) Auditory-Perceptual Space.
2. "run_dnn_categorization" - an example of a DNN categorization model building pipeline. It uses the same dataset of Cardinal Vowels as above.

Additionally, there is also a "settings.py" file where various parameters can be easily adjusted. More examples are provided in the "settings_examples" folder.


#### References:
Miller, J. D. (1989). Auditory-perceptual interpretation of the vowel. *The Journal of the Acoustical Society of America*, 85(5), 2114–2134.

