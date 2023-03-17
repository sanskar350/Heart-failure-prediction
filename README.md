# Heart-failure-prediction
This Python notebook aims to predict heart failure based on a dataset of various clinical and demographic features. The notebook uses machine learning algorithms to classify patients as having either high or low risk of heart failure based on their individual characteristics. 


# 1. Part 1:
This section is focused on the front matter, problem definition and problem evaluation with the
objective of providing solutions using AI applications to predict risk of having cardiovascular disease.
In this part we have chosen and proposed to work on critical healthcare chronic condition/healthcare
problem and identity and determine if the patient will develop cardiovascular disease or not in
upcoming years by using python, tensorflow and given data sets.
1.1 Problem Definition
Predication of risk of developing chronic Heart disease
This study is to determine whether a patient has risk of having coronary heart disease in future.
Coronary Heart disease is the condition when heart is not able to work properly as it works in normal
condition because of narrow or stretched arteries, clotted arteries or with heart structural problem
which make disturbance or stop normal blood flow.

## 1.2 Problem Evaluation

### Types of Heart Disease: -

Coronary Artery Disease:- This is the most common type of heart problem. It generates the problem
by building up clots or plaque in major blood vessels. Plaque could be generating by cholesterol or
other sustains which is taken in by a person via food, smoking, liquor etc. Resultant of plaque or clot
reduces blood supply in heart and other arteries. Heart gets less oxygen and less nutrient. This cause
creates the risk of cardiac arrest.
Congenital Heart Defects: - This is a condition when a person has a heart structural defects by birth.
Like Abnormal heart valves, Hole in heart chamber or in heart wall, may be a cause of missing a heart
valve.
Arrhythmia: This heart problem generates the irregular heartbeats rhythms. When heart beats
irregular norms like rapid heartbeats, slow heartbeats, early heartbeats, erratically or disturbed
heartbeats. Person can feel this problem because of abnormal heart function. It can be detecting by
ECG Electrocardiogram.
Dilated Cardiomyopathy: - When the person’s heart muscles get stretched or get thinner by toxins.
Previous heart attack or irregular heart function. Weak heart is not able to pump the blood in all the
chambers and it can cause a heart failure.
Myocardial Infarction : In common words it is known as Heart attack . When interrupted blood flow
damaged a part of working heart muscles. Then damaged part stop participating or start less
contributing in the heart function. Then person get a heart attack and the main reason of this
interruption could be a plaque, narrow artery or deposit of cholesterol in arteries.
Heart Failure: - Heart failure is a condition when the heart works but not like a healthy heart. In this
case the frequently high blood pressure and artery disease affects the heart capacity to pump blood.
This can be treated seriously because it can also generate complex conditions.
Hypertrophic Cardiomyopathy:- This problem developed by genetic problem. If the person has any
family history of heart disease, so person should take rescanning and need to diagnose the problem
before the time. Person has to be careful before its get worst with the time unknowingly. Heredity
can continue its culture so that person has to start taking precaution and healthy work life.
Mitral Valve Regurgitation : When the heart valve didn’t work well .open and closing process of the
valve could not be functioned properly and generate the pressure on veins and increase the blood
pressure. This can be a cause of heart failure. For this physician treated this problem by giving
medicines of blood thinner. It reduces the pressure of blood in veins and avoid the heart failure
condition.
Aortic Stenosis : When the mitral valve is working accordingly ,when it doesn’t close properly and
provided a restricted blood flow path. Restricted blood flow could be creating several problems
which led the problem of heart attack. 

### Symptoms of Heart Failure: -
Chest Pain- This is the primary and major symptom. Person who is suffering or going to have a heart
disease will have some heaviness or chest pain in canter part of chest is very obvious sign.
Breathing Difficulty- Most of the time patient or person of heart disease found difficulty in breathing.
Swelling in Ankle and other body parts - this one is the most common symptom of heart disease.
Interrupted blood flow in veins generates less blood flow in other body parts and it can be a cause
of swelling.
Heart Palpitation- Irregular heart function led the abnormal palpitation. In that case heartbeats
changes it normal momentum and start beating in fast, slow and early heartbeat rate.
Fatigue- Fatigue is the condition where less blood flow and fewer nutrients are not able to reach all
the body parts. In this condition person starts feeling less active and sleepy.
Abnormal heartbeat- In coronary heart disease plaque and clots of cholesterol and toxins subtracts
block the path of blood and provide a narrow path for blood flow. This process build up pressure on
the veins and make pressure on heart to force the blood to make its flow normally. This process
makes the heartbeats abnormal where in some part of the blood flows in normal way on the other
part or in some artery blood flows with pressure.
Swatting: - Person with heart disease easily get tired and start swatting because of its abnormal heart
function oxygen and nutrients were not able to supply enough energy and oxygen for workout, so
person get easily gasp and swatting.
Numbness: - Most common symptom of heart stroke. Here the particular body part or a side of body
leg, hand or mouth start feeling numbness or less active. Sometime Person couldn’t sense the body
part. Where he is not able to move or work with that particular body part.
Fainting or Unconsciousness: - lack of energy and loss of balance generates this symptom.
Cause of Heart Disease :- There are various reason of heart attack such as High Blood Pressure, High
Cholesterol/ other Lipids, Smoking/ Tobacco, Harmful use of Alcohol, Over stress, Anxiety,
Hypertension, Uncontrolled Diabetes, Obesity, Genetic problems, Age concern, Unfit or low activity,
Unhealthy Food hobbit, Unhealthy routine, Medicines Side effects, Respiratory system failure,
Structural Defects by birth, Chocked Arteries, Nutrition deficiency, Kidney disease, Improper Sleep,
Quality of Care. (Adam Felman, Sept 2020) [1]

## Motivation
As per World health organisation “Cardiovascular disease (CVDS) are the number one cause of death
globally, taking an estimated 17.9 million lives each year.’’

World health Organisation is the one the most active and authorised organisation which is working
globally and track records of all the medical and health related matters. As per Worldwide statistics
data provide the information that 17.9 million or 31% of death reason is cardiovascular heart disease.
Where 75% of the deaths data is gathered from low or middle class developing countries. Developing
countries are those countries which are less developed in industrial and financial norms. In these
countries human development index is poor means there are not appropriate arrangements for
natives. Natives are suffering from poverty, less medical facilities, hunger, lower wages, etc. (Anon.,
May 2017) [2]
According to the statics 85% of all the heart disease patients died with heart strokes or cardiac arrest.
These data are featuring bitter truth that the condition is worst in terms of heart disease. People are
aware about the heart disease but there are not aware how to handle that condition.
Health science and modern techniques are now started working on it. Some of major health science
institutes are working and developing predefined aspects of this problem. They use data set of
patients and set some parameters. Hence these techniques have fixed formulas they input the data
and analysis and diagnose the correct problem. These techniques are not only just diagnosing the
actual problem it also suggests the correct treatment as output.
Techniques are available but their sources of implementation are not that easy. These techniques
have to provide worldwide facilities with healthcare industries or with the government. Government
have to improve its own counties healthcare management and cover the each and every part of it.
By these efforts’ countries will be able to collect all the data of their natives and generate a complete
data source. This data source will inform the countries full index of healthy and unhealthy person. It
can predict the person’s health condition and future health related issues. These data source
generate a full report of individual and define it with authentication. It will also suggest the full
treatment as an output. (Anon., n.d.) [3]
By using these techniques and improving healthcare industries, healthcare worker also can diagnose
and treat the patient well on time. Successfully treatment can reduce the number of death cases.
Heart Disease is one of the major problems to solve. If we are taking initiatives to solve the problems.
Heart disease can be predict and diagnose on time and it can be treated as well, so the death rate
could be decrease over the time. Heart disease can be preventing and control in mortality rate.
Current situation is much more difficult than Normal situation. Due to COVID-19 . Scenario is
different and much more complicated. COVID-19 mortality rate percent increased. A heart patient
has to more careful and should protect himself from COVID-19.If you are infected with COVID-19
and you are a heart patient so it can more difficult condition for patient and healthcare workers to
treat and recover from illness. People of cardiovascular heart disease has higher death rate in Covid19.where COVID-19 Virus can begin building up blood clots in arteries and it can create a situation
of swelling in artery walls that can be a cause of heart attack. So one should not delay to contact the
concern physician if they find any symptoms of heart disease unless you are recovered from COVID.
Symptoms are mentioned above in first point.
Proposed AI and Machine Learning applications support Healthcare and provide a better way to
collect data and less time-consuming solutions. As we know medical staff cannot handle and give
proper attention to the patients in COVID-19 scenario. So, the proposed technique is helpful not only
for healthcare workers but also for patients who are getting the right treatment at the right time. It
can be a lifesaving application. Somehow we can reduce the mortality rate of decreasing people by
heart attack during Covid outbreak.

## 1.3 Conclusion to part 1
In part one we have discussed on the problem definition and reason to choose this problem
to work on identity problem and related data set.

# 2. Part 2 : Methodology
In this part we will be implementing, testing, and comparing classifiers on a real-world problem, the
objectives of part 2 included investigating the effects of feature selection. This requirement led our
selection of a dataset which would allow us to draw a strong comparison between various models
constructed.
Having drawn up a comparison matrix to score the chosen datasets, it was observed that the “dataset
of Framingham heart study” consisted of data from a well-defined experiment with 15 clear output
classes and with the largest number totalling 4240 data sets and 15 plus the class column. It was
determined that this significant set of features would represent a challenge for the construction of
the certain models and would offer sufficient chance to prove performance improvement,
particularly predication accuracy.
There were many other factors which supported the choice of the dataset of Framingham heart
study dataset versus the others, including the almost balanced class attribute, the large number of
access to this data set (by the total views) and real time health care data used for research, which
mean it would possibly be likely to directly match our results with model constructed and brought
results by other researchers.
We both have also decided that which tools, techniques, methodology, models should be used, to
process and classify the data for given assignment. Python Data Analysis Library, Numpy is an
optimized Python library, Matplotlib, Seaborn, SKLEARN or scikit-learn, Tensor flow was chosen
based on experience and ease of accessto we both group members. This also gave us an opportunity
to spend time on agreed programming language, tools, and models.

## 2.1 Datasets
To work on this task mentioned in brief all classification models and Artificial Intelligence and
Machine Learning techniques are applied to real-time healthcare dataset that are freely available
online on https://www.kaggle.com. We consumed this dataset for different tasks: Classification,
regression and text classification.

### 2.1.1 Datasets Classification
Various sources of Healthcare datasets were found online but we decided to choose and go with
Framingham heart study dataset to apply various AI techniques. Having chosen to fully implement
and tested dataset further decided to populate with as realistic results as possible. The following
sources were used:
1. Kaggle dataset of Framingham heart study
2. This data set is available on public domain and initially accessed via the link in the assignment
brief:
https://www.kaggle.com/datasets
3. Data sets File name - framingham.csv
4. The dataset present on Kaggle.com, it is a result of ongoing coronary heart disease research
which is based on the residents of Framingham city in Massachusetts, USA. The primary objective
is to predict the 10-yrs risk of cardiovascular disease, based on the 4240 patient’s data we have.
This data sets is giving us patients information, 15 attributes/columns.
5. The information of the dataset are as below :
Demographics -
Sex : Male or Female
Age : The age of the patient/person
Education level : Education level of the patient
Behavioural -
Current Smoker : if the patient is a current smoker. 0 for no and 1 for yes
Cigs per day : number of cigarettes that person smoked
Medical history -
BP Meds : if the patient is taking blood pressure medication. 0 for no and 1 for yes
Prevalent Stroke : If the patient had a stroke before
Prevalent Hyp : if the patient has hyper tension. 0 for no and 1 for yes
Diabetes : If the patient had diabetes. 0 for no and 1 for yes
Other :
tolChol : Total cholesterol of the patient
sysBP : higher BP count of the patient
diaBP : lower BP count of the patient
BMI : Body Mass Index of the patient
hearRate : Heart rate of the patient
glucose : Glucose level of the patient
Target :
10 year risk of coronary heart disease (CHD) . 0 for no and 1 for yes

The above datasets were organised into a set of master CSV files covering all the data elements
required. This was further processed in Python to generate individual CSV files for the relational
schema design outlined above. Where necessary, the Python scripts created ID values and crossreferenced multiple files to populate key values
(Swamynathan, n.d.) [5]

# 2.2 Evaluation and experiments
Classification Algorithm
• Logistic Regression
• Decision Tree Classifier
• Random Forest
• k-Nearest Neighbors Algorithm (KNN)
• Naïve Bayes (NB)
• Support Vector Machines (SVM)
• Artificial Neural Network
Each model was tested using Python Data Analysis Library. We have used common python module
used in data processing and data analyzation. This allowed investigation to find the performance in
terms of classification accuracy.

### 2.2.1 Logistic Regression:
Description
The logistic regression model models the probability of a classification problem with two possible
outcomes. It is an extension of the linear regression model used for classification problems.
Interpreting weights in logistic regression is different from interpreting weights in linear regression,
because the result of logistic regression is a probability between 0 and 1. The weight no longer affects
the probability in a linear manner. The logistics function converts the weighted sum into a
probability. Therefore, we must reconstruct the equation for explanation so that there is only one
linear term on the right side of the equation.
(Molnar, 02-2019 ) [6]


### 2.2.2 Decision Tree Classifier

Description
Linear regression and logistic regression models will fail when the relationship between factors and
results is nonlinear or when factors affect each other. Now it’s time to show the decision tree! The
tree-based model splits the data several times based on some clipping values in the elements.
Splitting will create different subsets of the data set, and each instance belongs to a subset. The leaf
set is called leaf or leaf node, and the intermediate subset is called internal node or split node. In
order to predict the result at each leaf node, the average result of the training data at that node is
used. Trees can be used for classification and regression. A tree can be grown using different
algorithms. They differ in the possible tree structure (for example, the number of subdivisions per
node), the criteria for finding subdivisions, the time to stop subdivisions, and the evaluation of simple
models. Inside the leaf node. Classification and regression tree algorithm (CART) is probably the most
popular tree induction algorithm. We will focus on CART, but for most other tree species, the
explanation is the same. The following formula describes the relationship between the outcome y
and features x. (Molnar, 02-2019 ) [6] 

### 2.2.3 Random Forest:
Description
The Random Forest (RF) algorithm forms a series of classification methods based on the
combination of several decision trees. The specialty of this classifier set (EoC) is that their tree-like 
Artificial Intelligence: Group 14
14
components are grown from a certain degree of randomness. This RF idea is defined as the general
principle of a random set of decision trees. The basic RF unit (called the basic learner) is a binary
tree constructed using recursive partitioning (RPART).

### 2.2.4 k-Nearest Neighbours
Description
The KNN algorithm measures similarity among the input sample and training instances it is a
memory-based learner. Similarity is evaluated using a distance measure and the instance is classified
using majority vote between the K neighbours. The KNN method is an example-based learning
method in which all available data points (examples) are saved and similarity measures are used to
classify new data points. The idea behind the KNN method is to assign new unclassified examples to
the class to which most of its K nearest neighbors belong. When the number of samples in the
training data set is large, the algorithm is found to be very effective in reducing misclassification
errors. Compared with many other supervised learning methods (such as support vector machines
(SVM), decision trees, neural networks, etc.), another advantage of the KNN method is that it can
easily handle problems with a class size of three or more.

### 2.2.5 Gaussian Naïve Bayes Classifier:
Naive Bayes is a machine learning algorithm that we use to solve classification problems. It is based
on Bayes' theorem. It is one of the simplest and most effective machine learning algorithms found
in many industries.
Naive Bayes classifier is a supervised machine learning method based on Bayes' theorem. In short,
the naive Bayes classifier assumes that the existence of a particular attribute of a class has nothing
to do with the existence of another attribute of that class. It is usually used to calculate the rear
probability based on the observation data and make a decision with a higher probability.

### 2.2.6 Support Vector Machines (SVM)
Description
Support vector machine is generally considered as a classification method, but it can be used for
classification problems and regression problems. It can easily handle multiple continuous and
categorical variables. SVM creates a hyperplane in a multidimensional space to separate different
classes. Iteratively generate the best hyperplane, which is used to minimize errors. The main idea
behind SVM is to find the largest marginal hyperplane (MMH) that divides the data set into classes
to the greatest extent.

In our study we can conclude that the logistics regression technique gave us the best results for this
given dataset. The diagnosis of heart disease in most of the cases is highly dependent on complex
combination of medical and Lab investigation, family history of Hypertension and coronary artery
disease due to all these complexities, healthcare researchers are immensely interested in correct
prediction of heart disease.
• First Part- Factors that remains same-
• AGE- Heart disease risk gets increased by the age. According to a research 45year in male
and 55year in female and older have great risk of heart attack.
• SEX- According to research Heart disease risk is more in male as compare to women because
estrogens hormone provides some protein which works against the heart disease.
• Origin- According to a research different race or origin have different risk rate of heart
disease. This is because of the certain living condition and genetic ethnicity. Like south Asians
have high risk of heart disease than East Asians.
• Family History-Heart disease risk factor will increase if you have family history or have heart
disease in your blood relatives.
• Second part- Factors that can be control by the person
• By controlling Cholesterol and Triglyceride Fat level.
• By controlling High Blood Pressure.
• By controlling Body Mass.
• By controlling Diabetics.
• No Smoking and use of Tobacco.
• Limited Consumption of Alcohol.
• Avoid Over Stress
• Taking Proper Medication on time if you have an illness.
• By eating healthy and Balance Diet.
• By Good Management of Anxiety.

• Third Part:- If someone has already heart disease and it has to admit in hospital because of
heart stock ,then here I am suggesting some clinical and medical options. After having proper
treatment under the healthcare expert people can save his live.
• Medicinal Treatment
• By Using of Blood Thinner Medicine
• By using Ant platelet ASA(Acetylsalicylic acid) Medicines ,aspirin
• Beta Blocker Medicines to reduce high blood pressure
• Calcium Channel Blocker Medicine to relax the heart vessels.
• Cholesterol lowering Medicines To reduce the bad cholesterol from the blood
• Digitalis Medicine for Heart Support
• Diuretics Medicines to reduce the pressure on heart and remove the excess water
• Vasodilators Medicine To control Blood Pressure
• Surgical Treatment
• Coronary artery Bypass Surgery –In this surgery surgeon replaced or repairs the damaged or
clotted veins with the health veins. With this they create the path for normal blood flow.
• Valve Replacement- Surgeons replace the heart valve with the damaged or not working valve
to support the heart.
• Device Implantation- Surgeon implant Pace maker, Balloon Catheters in heart to regulate
the heart beats and to control the blood flow in heart.
• Laser Treatment- It can help to treat Angina (Chest heaviness or pain in centre of the chest
).When the surgeon cannot perform bypass surgery due to certain reasons. They took help
of laser treatment which treatment is called by TMLR (Tran myocardial Laser
Revascularization ).
• Maze Surgery- Surgeon will create a electronic path by heat and cold therapy to regulate the
heart beats properly.
• Heart Transplant- This one is the most difficult procedure. This surgery will take time perform
and the right donor for the patient is the most difficult task. (Kinman, Sept 2018) [4]
• By Doing Regular Exercise/ Activity
• By having Good rest and Sleep.


## Prevention and Recommendation
In this assignment we have used various algorithm and artificial intelligence technique for predicting
and perceiving future patients of heart disease, which can help healthcare professional to predict
the conditions of heart disease based on medical, social, behavioural, demographics of patients. The
data set used with study of the cardiovascular system of residents of Framingham, Massachusetts,
and includes more than 4240 data sets and 15 attributes can be used with other attributes that can
be included by the healthcare profession plus our algorithm will provide an accurate result. This
algorithm and classification method can be used to predict of coronary heart disease so that looking
predictions, Health care professional can take data driven decision to provide preventive actions to
patients for their heart disease. Here are some Heart Disease Preventions :

Heart Disease Prevention :- Heart disease is one of the major cause of death in the world. There are
many factors that can raise the risk of heart disease. Here I am mentioning some factors which are
divided in three parts. First parts contain those factors which cannot change by person. In the Second
part those factors are mentioned which can be changed or improved by individual. Third part
describes the clinically medication and suggestion.

# 3. Conclusion
This report has successfully completed the tasks as outlined in the assignment brief. Part 1 and part
2 covers all the Task and its respective sub-tasks. Specifically, starting from Problem statement, the
conversion into solutions, methodology and models used to experiments, results and discussion to
resolve the problems using Artificial intelligence and Machine learning technologies. The advantages
and challenges of each problem were explored using test data. The Framingham heart study dataset
was analysed and interpreted using Logistic Regression, Decision Tree Classifier, Random Forest, K
Nearest Neighbours, Naïve Bayes, Support Vector Machines, Artificial Neural Network classification
models. Each model was evaluated with a test /train split. Finally, the models were critically analysed
based on their performance in accuracy.

# 4. References
REFERENCES:-
[1] Everything you need to know about Heart Disease –Written by Adam Felman, Medically
Reviewed by Joyce-Oen-Hsiao, Published on Medical News Today updated on 29
September 2020.
[2] Cardiovascular Disease (CVDs) WORLD HEALTH ORGANISATION newsletter , Published
in 17 May 2017,www.who.int
[3] Data from U.K. Office for National Statics, Published on April16, 2020.
[4] Heart Disease Prevention medically Reviewed by Debra Sullivan, Ph.D., MSN, R.N.,CNE,
COI, -Written by Tricia Kinman – Updated on September 212018
https://www.healthline.com/reviewers/debra-sullivan-phd-msn-rn-cne-coi
[5] Mastering Machine Learning with Python in Six Steps Manohar Swamynathan, 2021,
Pages 1-52]
[6] Interpretable Machine Learning A Guide for Making Black Box Models Explainable
Christoph Molnar This book is for sale at http://leanpub.com/interpretable-machinelearning This version was published on 2019-02-21]
