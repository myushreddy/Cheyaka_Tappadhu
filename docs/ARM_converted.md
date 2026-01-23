# ARM: Adaptive Rank-Based Mutation for Android Malware Detection under Adversarial Attacks

**Authors:** Teenu S. John & Tony Thomas

**Published in:** Journal of Cyber Security Technology  
**ISSN:** 2374-2917 (Print) 2374-2925 (Online)  
**Journal homepage:** http://www.tandfonline.com/journals/tsec

**DOI:** 10.1080/23742917.2025.  
**Published online:** 18 Aug 2025  
**Article views:** 59

---

## Abstract

The escalating threat of Android malware has led to the widespread adoption of machine learning (ML) techniques for detection. While ML-based methods have demonstrated strong performance, they remain vulnerable to adversarial attacks designed to manipulate application features and evade detection. To address this challenge, we propose a robust and adaptive feature selection mechanism based on a Genetic Algorithm enhanced with Rank-Based Adaptive Mutation (GA-RAM). In this method, the mutation rate is dynamically adjusted according to the performance of feature subsets: lower-ranked subsets undergo more aggressive mutations to eliminate features that are likely to be injected by adversaries and are behaviorally irrelevant, while higher-performing subsets are preserved to retain important, semantically meaningful features. This adaptive balance between exploration and exploitation enables the model to converge on resilient and interpretable feature combinations. The proposed mechanism is evaluated under a comprehensive set of adversarial scenarios, including white-box (FGSM, JSMA), grey-box (mimicry, salt-and-pepper noise), black-box (GAN-based), and zero-day attacks. The system achieved 98.6% accuracy in detecting general malware and maintained over 90% accuracy across all adversarial attacks, including 94.1% for zero-day malware. Additionally, SHapley Additive exPlanations (SHAP) are employed to enhance the interpretability and trustworthiness of the detection process.

**Keywords:** Android malware detection; adversarial attacks; white-box attacks; black-box attacks; grey-box attacks

**Article History:**
- Received: 20 May 2025
- Accepted: 5 August 2025

---

## 1. Introduction

Nowadays, Android smartphone users are threatened by the growing malware attacks [1, 2]. In the year 2023, 600 million malicious applications evaded Google Play detection [3]. Most antivirus detection mechanisms employ traditional signature-based detection, in which the application's hash value is compared with the hash values of known malware. When a zero-day attack occurs, the signature-based mechanism cannot detect them [4]. With the revolution of machine learning (ML), cybersecurity practitioners and researchers are utilizing it for advanced cyber threat detection [5, 6]. In Android smartphone malware detection, ML based detection mechanisms gave promising results over the past decades. However, these mechanisms can be evaded by an adversary by using various adversarial attacks [7]. The recent study conducted by Kaspersky highlights that although machine learning-based antivirus (AV) systems offer powerful detection capabilities, they are highly susceptible to adversarial manipulation, emphasizing the need for robust defenses [8].

Several studies have explored adversarial attacks on malware detection systems in Android environments [9, 10]. Most of the existing adversarial malware are crafted by perturbing the static features. One common tactic is API call injection, where benign-looking API calls are inserted into the code without affecting its functionality. Likewise, attackers employ permission injection by adding benign or commonly requested permissions, which masks the presence of high-risk permissions. Other techniques are API reordering and injecting dead codes to skew statistical detection models to confuse sequence-based models by disrupting the order of significant API calls [11]. More advanced techniques include feature-space adversarial attacks, where malware is crafted using optimization or learning algorithms such as the Fast Gradient Sign Method (FGSM), genetic algorithms, etc [12, 13]. to deliberately mislead detectors by exploiting their decision boundaries.

Further, malware may also employ obfuscations such as identifier renaming, reflection, and string encryption etc. to evade detection. These adversarial attacks make static malware detection mechanisms ineffective, despite their benefits such as high code coverage and the capacity to identify run-time evasive malware. There are several adversarial defense mechanisms proposed in literature. Among them, adversarial training enhances robustness but depends heavily on large, diverse datasets and often overfits to known attack types, making it ineffective against novel adversarial strategies [14]. Although feature squeezing strategies [15] and input sanitization reduce the attack surface, they are likely to remove important behavioral features, leading to degraded accuracy or increased false positives. Graph-based methods on the otherhand capture structural and semantic relationships [16–18] but remain vulnerable to call graph obfuscation and incur high computational costs [19]. Ensemble learning mechanisms [9] improve resilience by combining multiple detectors, but are resource intensive and still susceptible to transferable adversarial examples. Therefore, a mechanism that is computationally less intensive and capable of combating various adversarial attacks has become the need of the hour.

While crafting the attacks, adversaries often prefer feature injections over feature removal, as eliminating core features may disrupt the malware's intended functionality or cause it to crash [20, 21]. The injected features are typically behaviorally irrelevant and do not contribute to malicious operations. The existing feature selection mechanisms require manual or heuristic grouping [12] to identify core malicious features which is labour intensive. Ensemble based feature selection on the other hand depends heavily on the choice and tuning of base classifiers, which may not generalize across datasets or malware families [22]. 

To address these challenges, we propose a Genetic Algorithm enhanced with Rank-Based Adaptive Mutation (GA-RAM) which iteratively evolves feature subsets based on their impact on detection accuracy. In the proposed mechanism, low-ranked chromosomes are subjected to more aggressive mutation, leading to significant alterations in their feature subsets. This increases the chance that irrelevant or features that are likely to be injected will be removed and new, potentially relevant features will be introduced. Feature sets with higher detection accuracy are assigned lower mutation rates, which ensures that effective and semantically meaningful feature combinations are maintained across generations. Over successive iterations, GA with RAM converges on robust and semantically meaningful feature sets by adaptively balancing exploration (through aggressive mutation of low-ranked individuals) and exploitation (by conserving high-performing solutions). The feature set that obtained maximum accuracy are then used to train the classifier for detecting various attacks. 

In the proposed mechanism, we utilize both API calls and permissions as feature sets for detecting adversarial Android malware. Since permissions requested by an app, declared explicitly in the Android manifest is hard to obfuscate [23], we used them as they can be used to detect obfuscated malware such as those employing string encryption and identifier renaming.

To evaluate the robustness of the proposed detection mechanism, we employed a range of adversarial attack techniques categorized under white-box, grey-box, and black-box threat models. In the white-box setting, we used FGSM and Jacobian-Based Saliency Map Attack (JSMA), which generate adversarial samples by leveraging the model's gradients to perturb features strategically. For grey-box attacks, we applied salt-and-Pepper noise and mimicry attacks. In the black-box setting, we used Generative Adversarial Network (GAN) based attacks, which generate adversarial samples without the knowledge of the model architecture or parameters of the targetted detection model. We also evaluated our mechanism for detecting zero-day malware [24]. Further, we use SHAP [25] to explain the rationale behind the accuracy of our mechanism. 

### Contributions

The contributions of this work are:

- **A novel feature selection mechanism, GA-RAM,** is proposed that enhances Android malware detection by adaptively evolving feature subsets through rank-based mutation, effectively identifying and removing features that are likely to be adversarially injected and behaviorally irrelevant while preserving semantically meaningful features.
- **The proposed mechanism was comprehensively evaluated** across multiple adversarial threat models including white-box, grey-box, black-box, and zero-day attacks and demonstrated high detection accuracy and robustness, with its predictions further supported by SHAP-based interpretability.

The remainder of this paper is structured as follows: Section 2 reviews relevant literature in the field. Section 3 presents a comprehensive overview of adversarial attacks. The proposed methodology is detailed in Section 4. Experimental results are reported in Sections 5 and 6 discusses limitations and future work and Section 7 concludes the paper.

---

## 2. Related Works

This section details the existing works in Android malware detection and also the adversarial attacks and defense mechanisms implemented till date. The three types of Android malware detection mechanisms are static, dynamic and hybrid. In static mechanisms, the malware is identified with the static features, without executing the application. Dynamic detection mechanisms are those in which the application is made to execute in an isolated environment for extracting its runtime features for examining malicious activities [26]. Hybrid mechanisms on the other hand combine both of these for malware detection. The following sections analyze the existing general and adversarial Android malware attack and detection mechanisms.

### 2.1. General malware detection mechanisms

Static detection mechanisms take permissions [27, 28], API calls [29, 30], opcodes [31, 32] and other static features for malware detection. Many static malware detection mechanisms rely on computing the similarity between the target application and known benign or malware applications to identify malicious software. However, this approach often leads to high false positive rates, as some legitimate applications may share features commonly associated with malware. There are many deep neural network-based malware detection mechanisms that gave promising results [33–36]. However, the complexity associated with training deep neural network is high. In [27], the mechanism uses permission pairs to detect malware. However, the false positive rate of the mechanism increases significantly due to the occurrence of the same permission pairs appearing in legitimate applications.

To detect malware that executes at runtime, several dynamic mechanisms are also proposed [37, 38]. In [39], an ensemble learning mechanism was proposed that used the system level features to detect malware. In [40] battery usage, running processes etc. are used to detect malware. But, their mechanism was not tested on real malware datasets. In [41], the mechanism used layout graphs obtained from user interface traces to detect malicious behaviour. However the mechanism fails if the application has minimal user interface traces. In RepassDroid [42] the mechanism used API call graph of the application to detect malware. But, the effectiveness of the mechanism for detecting adversarial attacks are not evaluated. There are several mechanisms that used frequencies of system calls to detect malware [43]. However, when an app is repackaged it may contain certain system calls that are frequently seen in legitimate application causing misclassification. In [44], a malware detection mechanism with Long Short Term Memory(LSTM) model with system calls as features was proposed. However, the lack of preprocessing may induce noise in the detection mechanism.

There are various hybrid malware detection mechanisms that utilize the advantages of static and dynamic mechanisms [45, 46]. In [47], a hybrid detection mechanism was developed. But the effectiveness of the risk assessment mechanism under various attack scenarios was not tested. In HADM [48], a hybrid detection mechanism was proposed with SVM.

There are a number of challenges associated with these malware detection mechanisms. Static detection methods struggle with high false positives [27], while dynamic methods are hindered by reliance on synthetic datasets and incomplete feature extraction, such as user interface traces. Hybrid models, though combining the strengths of both, increase computational overhead and often lack resilience against adversarial attacks.

### 2.2. Adversarial attacks and detection

This section outlines existing adversarial Android malware attacks and their corresponding defense mechanisms.

#### 2.2.1. Adversarial attacks

In [49], Pierazzi et al. crafted adversarial malware by adding code transplantations to evade detection. Bala et al. [20] proposed a poisoning attack by mutating API calls and permissions to evade detection. In [50], a semiblackbox attack against deep learning-based Android malware detection using simulated annealing algorithm for evasion was proposed. Xu et al. [13] proposed an algorithm using genetic algorithm and used attention mechanism and JSMA to craft adversarial samples. Li et al. [51] proposed an adversarial attack based on a bi-objective Generative Adversarial Network (GAN) to evade both Android malware detection systems and (firewalls).

In conclusion, the rise of adversarial attacks, such as problem space, poisoning, and semiblackbox attacks, has exposed significant vulnerabilities in existing Android malware detection mechanisms. Techniques like modifying code, mutating API calls, or leveraging algorithms such as simulated annealing and GANs have demonstrated the ability to bypass current defenses.

#### 2.2.2. Adversarial detection

There exists several adversarial Android malware detection mechanisms [52, 53]. Most of them employ adversarial retraining [54, 55], adversarial feature selection [12, 56, 57] and ensemble learning [58]. In [59], the authors proposed a feature selection mechanism with deep reinforcement learning and recurrent neural networks. However, implementing these methods for feature selection requires significant computational resources. Moreover, interpreting the reasoning behind deep reinforcement-based feature selections is difficult. In [60] the authors proposed DANdroid, an adversarial learning mechanism. Their mechanism was only tested for obfuscated attack detection. In [61] the mechanism used adversarial training. But the generated samples were not tested to validate whether they preserved the malicious nature. In [62], a mechanism called SecCLS was proposed that used a feature selection mechanism. Their mechanism was not tested for detecting mimicry attacks. In [63], the authors proposed an ensemble learning mechanism to detect adversarial attacks. But their mechanism was evaluated only for ransomware. In [64], a mechanism was proposed to detect adversarial malware with GAN. However, their mechanism was not tested with diverse adversarial samples. In [65], the mechanism proposes adversarial training to detect adversarial samples. However, training causes high overhead and can only detect a few adversarial attack types. In [14], the mechanism proposes q-learning to detect adversarial samples. However, their mechanism only considered permission injection. In [66], various ML models are evaluated to detect zero day attacks in Android. In [67], the authors combined zero shot learning with deep learning to detect Android malware, but yielded low performance.

### 2.3. Zero day Android malware detection

In [68], a zero day malware detection mechanism was proposed by using control flow graphs. However, feature injections can alter the control flow of the application. In [69], the authors proposed a multiview deep learning mechanism. However, their mechanism was evaluated on older datasets, and failed to evaluate the accuracy of the model on latest Android malware samples. In [70], the mechanism proposed a mechanism that used static features to detect zero day malware. However, identifying critical malware features among many features remains a challenge. In [71], the authors proposed that, deep learning models for zero day malware detection suffer from lack of interpretability reducing the trust of the prediction.

---

## 3. Adversarial attacks in Android malware detection

Adversarial attacks in machine learning (ML) are generally classified into white-box, black-box, and grey-box categories, based on the level of knowledge an attacker has about the target model [72]. In a white-box attack, the adversary possesses complete access to the model's architecture, parameters, and gradients [73–75]. This allows for precise and targeted perturbations of input features, typically using gradient-based methods such as FGSM or the JSMA, which craft adversarial inputs by maximizing the model's loss while keeping changes minimal and imperceptible.

In contrast, black-box attacks occur when the adversary has no information about the internal details of the model, such as its architecture or gradients [76]. Instead, the attacker probes the model by submitting inputs and analyzing the resulting outputs to infer decision boundaries and manipulate features accordingly [77]. GAN-based attacks are such types of attacks [78].

Grey-box attacks represent an intermediate threat model, where the attacker has a few information of the system, such as limited access to training data, the feature space, or the model's outputs. Two prominent examples of grey-box, gradient-free attacks are salt-and-pepper noise and mimicry attacks [9]. In salt-and-pepper attack, attackers perturb malware feature by adding noise in the form of irrelevant feature. Mimicry attacks [9], on the other hand, modify malware samples so that their feature representations closely resemble those of benign applications. Since both methods rely on knowledge of the input feature space and access to output feedback but not to internal parameters, they are characterized as grey-box attacks.

In this paper, we create adversarial malware using FGSM, JSMA, GAN, to evaluate the resilience of our proposed detection mechanism under diverse threat models. We also test the detection performance of the proposed mechanism in the presence of salt-and-pepper and mimicry attack samples of [9] and zero-day Android malware.

### 3.1. Mathematical formulation of adversarial attacks

Consider an application, which can either be benign or malicious. Let **x = (x₁, x₂, ..., xₙ)** denote its original feature vector, where **xᵢ ∈ {0, 1}** represents the presence (xᵢ=1) or absence (xᵢ=0) of the ith feature. Let **y ∈ {0, 1}** be the label of the application, where y=0 denotes a benign sample, and y=1 denotes a malicious sample. The aim of the adversary is to evade detection by various feature injections, by preserving core malicious features. To simulate the attacks, we flip features from 0 to 1 rather than from 1 to 0. The adversarial attacks simulated in our proposed mechanism are crafted as follows.

#### 3.1.1. FGSM

In this paper, we use binary, additive-only variant of the Fast Gradient Sign Method (FGSM). This is computed as:

**x_adv = max(x, x + ε · sign(∇_x J(X, Y)))**

where **x_adv** denotes the adversarial feature vector generated from the original feature vector **x = (x₁, ..., xₙ)**, **ε** is the perturbation magnitude, **∇_x J(X, Y)** is the gradient of the loss with respect to the input features.

#### 3.1.2. JSMA

In this paper, we adopt the binary, variant of the Jacobian-based Saliency Map Attack (JSMA), where only inactive features (xᵢ=0) are modified. The attack computes a saliency map that identifies the most influential input features for misclassification. The saliency map is defined as:

**S(X, 0)[i] = ∂F₀(x)/∂xᵢ - ∂F₁(x)/∂xᵢ**

where **F₀(x)** and **F₁(x)** denote the classifier's output probabilities for classes y=0 and y=1, respectively. The adversarial feature vector **x_adv** is then generated by modifying the most salient inactive feature:

**x_adv[i] = 1** where **i = argmax S(X, 0)[j]** for **j : x[j] = 0**

This process is repeated iteratively, modifying one feature at a time, until misclassification is achieved or a predefined perturbation limit is reached.

#### 3.1.3. Mimicry and salt-and-pepper

Mimicry attacks aim to modify a malware sample such that its feature distribution closely resembles that of a benign application. This is achieved by selectively activating features present in benign samples. On the other hand, salt-and-pepper attacks introduce noise into the feature space by randomly adding benign or irrelevant features (e.g. via dead code insertion) to a malicious application. Both strategies attempt to evade detection by shifting the malicious sample toward the benign region in the feature space. The adversarial feature vector **x_adv = (x₁_adv, x₂_adv, ..., xₙ_adv)** is generated by perturbing the original feature vector **x = (x₁, x₂, ..., xₙ)** as follows:

**x_adv[i] = max(x[i], noise[i])**

subject to the constraint **|x_adv| > |x|**, where **|x| = Σᵢ₌₁ⁿ xᵢ** denotes the Hamming weight of the feature vector. This increase in Hamming weight reflects the additive nature of the perturbation, where inactive features (xᵢ=0) are flipped to active (x_adv[i]=1) to fool the classifier.

#### 3.1.4. GAN-based attacks

In GAN-based attacks, a Generative Adversarial Network (GAN) is used to create adversarial malware. The GAN consists of a generator G and a discriminator D, which are trained in a minimax game. The generator learns to produce adversarial feature vectors resembling real malware, while the discriminator learns to distinguish real malware from generated ones. The training objective of the GAN is defined as:

**min_G max_D E_{x~p_data}[log D(x + η)] + E_{z~p_z}[log(1 - D(G(z) + η'))]**

where **x~p_data** is a real malware feature vector, and **z~p_z** is a noise vector sampled from a standard normal distribution. The generator **G(z)** produces adversarial malware samples, while the discriminator **D(x)** outputs the probability that input x is a real malware. Small Gaussian perturbations **η, η' ~ N(0, σ²)** are added to real and generated samples, respectively, to stabilize training.

All these adversarial attacks provided above result in:

1. **Feature Expansion:** The extended feature set includes new features that were not originally present.
2. **Critical Feature Preservation:** Despite the perturbations, critical features necessary for malware functionality, such as specific APIs or permissions, remain unchanged.
3. **Classifier Evasion:** These perturbations are crafted to shift the classifier's decision boundary, causing the malicious sample to be classified as legitimate.

---

## 4. Proposed Method

![Figure 1: Workflow of the proposed mechanism](images/workflow.png)

Figure 1 shows the workflow of the proposed method. After extracting the API calls and permissions from the Android applications, the proposed method combines Mutual Information (MI) and the GA-RAM to detect general, adversarial and zero-day malware by addressing their characteristic feature perturbations. MI identifies an initial feature set by evaluating the mutual information between Android application features, such as APIs and permissions, and their corresponding labels (benign or malicious). GA-RAM then refines this set using evolutionary principles to generate an optimized subset that effectively detects adversarial perturbations.

### 4.1. Mutual information based feature selection

The feature space of Android applications is large [79] and it is essential to compute the relevant features for malware detection. In Android, each API call corresponds to a specific function that the application can evoke such as accessing the internet, reading files to more sensitive functions such as retrieving the user's location, sending SMS messages, accessing the camera etc. Each of these functionalities are protected by specific permissions. As new features are introduced in Android updates, additional permissions are created expanding the permission space making it difficult to detect the most relevant features for malware detection. Hence we use Mutual information (MI) to identify the relevant APIs and permissions that can detect malware. MI filters out the most relevant features by computing their dependency on the target class, ensuring that the chosen features have a strong ability to distinguish between benign and malicious samples. MI has shown excellent results for identifying general malware and adversarial malware in Android applications [80, 81]. The mutual information is computed as follows.

Let **X = {x₁, x₂, ..., xₘ}** denotes the set of features extracted from an Android application, and let **Y = {y₁, y₂, ..., yₘ}** represents the corresponding labels, where **yᵢ ∈ {0, 1}** indicates whether the application is benign (yᵢ=0) or malicious (yᵢ=1). The mutual information **I(X;Y)** measures the dependence between the features X and the labels Y. This metric is utilized to identify the relevant features that are instrumental in identifying between goodware and malware. The mutual information is computed as:

**I(X;Y) = ΣΣ p(x,y) log(p(x,y)/(p(x)p(y)))**

where **p(x,y)** is the joint probability mass function of the feature x and its label y, **p(x)** and **p(y)** are the marginal probability mass functions of x and y respectively. The k-best features are obtained by maximizing the mutual information gain as follows. Let **S** be any non empty subset of **{x₁, x₂, ..., xₘ}** and **Y_S** denotes the sets of corresponding labels. The best feature subset **S** is determined as:

**S* = argmax_S I(S;Y_S)**

When **I(S;Y_S)** is high, it indicates that the feature subset **S** is highly relevant for distinguishing a malware from a benign application. The k best features obtained using the mutual information are given as the input to the GA-RAM.

### 4.2. Genetic algorithm with rank based adaptive mutation for feature selection

The relationship between APIs and permissions in Android can be used to identify the behaviour of the applications. For instance, if an app wants to send an SMS, it would use an API call such as **SmsManager.sendTextMessage()**. However, this API is protected by a permission called **SEND_SMS** to ensure that only apps that have been granted the specific permission can invoke this. Another example of a potentially malicious API and permission combination is the **READ_PHONE_STATE** permission and the **TelephonyManager.getDeviceId()** API. Likewise, malware should contain certain permission pairs for a successful attack [27]. For instance, certain permission pairs such as **INTERNET** and **READ_CONTACTS** is used by the spyware to read the user's contact list and send that data to a remote server. During the creation of adversarial malware, these critical permission pairs and API-permission combination often remain unaltered. This is because altering such essential features would likely compromise the core functionality of the malware. The following section explains how GA-RAM helps to identify critical features with its evolutionary operations.

#### 4.2.1. Initializing the feature sets and computing the fitness

In order to detect malware, the initialization step in GA-RAM generates an initial population of feature subsets, each representing a combination of 155 features (APIs and permissions) pre-selected using mutual information. A '1' in these subsets indicates that the corresponding feature is selected, while a '0' means it is not. The algorithm initializes 50 random feature subsets, allowing the GA-RAM to explore diverse combinations of features. By using accuracy as the fitness criterion, the GA-RAM iteratively refines the feature sets over multiple generations. Feature subsets that achieve the highest detection accuracy are selected using Tournament selection [82]. Tournament selection is a selection process commonly used in genetic algorithms where a group (or 'tournament') of feature subsets is first randomly selected, and then the best-performing subset based on its accuracy. The accuracy of the feature subsets are tested with general malware datasets [83, 84].

#### 4.2.2. Generating feature subsets that contain malware functionality

After the tournament selection, the crossover operator is applied to the feature subsets to combine features from different subsets to generate new ones. In the context of adversarial malware detection, this process is critical for creating robust feature combinations capable of detecting sophisticated attack strategies. For example, if one feature subset contains API-related features and the other contains key permissions, the crossover operator merges these subsets, resulting in a new feature subset that incorporates both API and permission-related features.

However, existing single-point crossover mechanisms commonly used in GA for Android malware detection [85, 86] are less effective in this context. Single-point crossover tends to preserve the sequence of features on one side of the crossover point, limiting the exploration of diverse feature combinations [87]. This lack of diversity can reduce the algorithm's ability to detect adversarial malware, which often requires the identification of interactions between critical API's and permission combinations.

![Figure 2: Accuracy values of single point and two point crossover mechanisms](image-placeholder)

Our proposed mechanism employs a two-point crossover operator, which significantly enhances the construction of critical feature subsets that adversaries tend to preserve. The 2-point crossover operator enhances the exploration of diverse feature combinations by selecting two points in the parent feature subsets and swapping the features between these points to generate new subsets. This approach allows features from different regions of the parent subsets to combine, resulting in greater diversity in the offspring feature subsets. In the context of adversarial malware detection, this diversity is particularly valuable as it enables the mechanism to capture complex interactions between critical API-related and permission-related features.

#### 4.2.3. Rank based adaptive mutation

After the 2-point crossover, the resulting feature subsets contain highly relevant features that contain relevant permissions and API calls. However, applying uniform mutation to these subsets may cause some '1's to change to '0's, potentially leading to the loss of critical features. To address this challenge, we propose to employ Rank based Adaptive Mutation (RAM) [89]. Unlike uniform mutation, which applies a fixed mutation rate on all the feature subsets, rank-based adaptive mutation adjusts the mutation rate based on the fitness ranking of each feature subsets. In this approach, the ranking is assigned such that the first rank corresponds to the feature subset with the lowest accuracy, which is subjected to the maximum mutation probability **p_max**. The remaining feature subsets are arranged in ascending order of their fitness, with the best-performing subset receiving the last rank. The subset with the highest accuracy is assigned a mutation probability of 0, ensuring that critical features that showed optimal performance are preserved [90].

In the proposed approach, chromosomes with lower fitness are subjected to more aggressive mutation, resulting in substantial changes to their feature compositions. This increases the likelihood of eliminating irrelevant features while introducing new, potentially informative ones. Conversely, feature sets that yield higher detection accuracy undergo minimal mutation, thereby preserving effective and semantically meaningful combinations across generations. Through successive iterations, the GA enhanced with Rank-Based Adaptive Mutation (RAM) progressively converges toward robust feature sets by striking a dynamic balance between exploration by intensified mutation of poorly performing individuals and exploitation by retaining high-performing solutions.

Let **p_max** denotes the maximum mutation probability. Suppose there are **n** feature subsets which are arranged in the increasing order of accuracy (in detecting malware). The mutation probability **pᵢ** for the ith feature subset is computed as:

**pᵢ = p_max × (i-1)/(n-1)**

where **n** denotes the total number of feature subsets. In the next generation, the newly mutated feature subsets are evaluated based on their fitness, which is typically measured by their classification accuracy. The GA-RAM in this way selects the feature subsets with the best performance, prioritizing those that preserve critical permission pairs and API-permission combinations, while discarding or further mutating underperforming subsets. As this process continues across multiple generations, the GA-RAM iteratively improves the population of feature subsets.

### Algorithm 1: GA-RAM

The algorithm works as follows:

**Step 1: Initialization**
- The algorithm begins by generating an initial population P of size Ps=50. Each feature set fᵢ within this population is represented as a binary vector containing Nf features where Nf=155.
- The best accuracy observed so far, A_best, is initialized to 0. The stopping counter, C_stop, is also initialized to 0, with a stopping threshold T=5.
- The genetic algorithm parameters are set with crossover probability cr=0.7 and maximum mutation rate p_max=0.6.
- Each feature sets accuracy is evaluated using a Random Forest classifier with an 80:20 train-test data split.

**Step 2: Fitness Evaluation**
- For each feature set fᵢ ∈ P, the accuracy Aᵢ(fᵢ) is computed to evaluate its ability to classify benign and malicious applications.
- The accuracy metric measures the quality of fᵢ in distinguishing between benign (y=0) and malicious (y=1) applications.

**Step 3: Selection**
- Using tournament selection, a subset of feature sets Ft = {ft₁, ft₂, ..., fts} is selected from the population P.
- This ensures that feature sets with higher accuracy have a greater chance of being selected for crossover.

**Step 4: Crossover**
- Two-point crossover is applied to the selected feature sets Ft with crossover probability cr, producing offspring feature sets Fc.
- Crossover combines features from two parent sets to explore new combinations of features.

**Step 5: Accuracy Evaluation of Offspring**
- For each offspring feature set fcⱼ ∈ Fc, the accuracy Aⱼ(fcⱼ) is computed.
- This evaluates the fitness of newly generated feature sets for malware detection.

**Step 6: Sorting and Ranking**
- The offspring feature sets Fc are sorted based on their accuracy, resulting in Fc'.
- Each feature set fc' ∈ Fc' is assigned a rank, where the most accurate feature set has rank 1.

**Step 7: Rank-Based Adaptive Mutation**
- The mutation rate p(fc') for each feature set fc' is computed using the formula above.
- Higher-ranked feature sets undergo smaller mutations, while lower-ranked feature sets are mutated more significantly to explore diverse solutions.
- Each feature set fc' is mutated to produce fc'', introducing new variations in the population.

**Step 8: Updating the Population**
- The maximum accuracy in the current generation, A_gen, is identified.
- If A_gen > A_best, the best accuracy A_best is updated, the stopping counter C_stop is reset to 0, and the population P is updated with the mutated feature sets fc''.
- Otherwise, the stopping counter C_stop is incremented.

**Step 9: Stopping Criteria**
- The algorithm continues until the stopping counter C_stop reaches the threshold T or the maximum number of generations Ng is reached.
- When the algorithm terminates, the feature set with the highest accuracy max(Aⱼ(fc'')) is returned.

### 4.3. Detection mechanism using ML

After identifying the features with the proposed feature selection, we train a machine learning model (e.g. Random Forest, Support Vector Machine, or Neural Network) using the optimized feature set shown in Figure 1. The model performs classification, identifying the application as either benign or malicious.

---

## 5. Experimental Results

The experiments were conducted on a Windows 10 machine equipped with an Intel i7 processor. The implementation was carried out using Python.

### 5.1. Dataset

One thousand five hundred malware applications from the Drebin dataset [84] and 3500 malware applications from the AndroZoo dataset [83], and 2000 malware applications from Virusshare were collected. All these malware samples emerged in the year 2012–2019. Additionally 63,000 benign applications were taken from the Google Playstore [3]. The categories of goodware applications used for training are provided in Table 1. We used more benign applications than malware applications to reduce the spatial bias for malware detection as mentioned in [91]. The benign applications were verified using VirusTotal [92] to ensure they were not malicious.

#### Table 1: Benign application categories

| No | Category | Number of Applications |
|----|----------|----------------------|
| 1 | Education | 7,330 |
| 2 | Entertainment | 6,500 |
| 3 | Business | 8,543 |
| 4 | Fitness | 6,345 |
| 5 | Games | 7,404 |
| 6 | Books and Magazines | 9,005 |
| 7 | Weather | 8,108 |
| 8 | Others | 9,765 |

To reduce the temporal bias, for testing, we collected 2500 malware samples from VirusShare [88] and Malwarebazaar [93] that emerged in the year 2020–2022. All the samples were verified with VirusTotal [92] to check whether they exhibited malicious behaviour.

For adversarial attacks, we used the feature peturbation method as mentioned in Section II. We used 1500 malware samples to perturb for FGSM, JSMA and GAN attacks. The malware samples were collected from the Drebin [84] dataset. For salt and pepper attacks, we used adversarial samples of [9]. For mimicry attacks, the dataset creation followed a 'mimicry × 30' approach, where 30 benign applications were selected. Malware features were injected into each benign application, and the sample requiring the least amount of perturbation was chosen for evaluation [9]. We used an adversarial dataset comprised of 250 mimicry attack samples, 243 salt-and-pepper noise attack samples, along with 1500 benign samples for testing.

To evaluate the effectiveness of the proposed detection mechanism in identifying zero-day attacks, we used 1500 latest malware samples from the Virusshare dataset [88] that emerged in the time frame 2023–2024. For testing, we included 1500 benign samples.

### 5.2. Feature extraction and feature selection

Using the Androguard tool, we extracted API calls and permissions from the collected applications. These features were input into the proposed feature selection mechanism, which selected 48 features. The proposed mechanism was evaluated on different machine learning models, and the performance is detailed in Table 4. Among the various ML classifiers, the Random Forest model achieved the highest accuracy. The Random Forest classifier was implemented with sklearn, using 100 decision trees. The fitness was computed using the accuracy obtained after training and testing with the general malware datasets.

#### Table 4: Performance of the proposed GA-based feature selection on different ML models

| ML models | Precision | Recall | Accuracy |
|-----------|-----------|--------|----------|
| Random Forest | 98.4% | 98.8% | 98.6% |
| SVM | 96.5% | 97.2% | 96.8% |
| Naive Bayes | 93.6% | 96.5% | 95.0% |

### 5.3. Performance evaluation

This section presents a comprehensive evaluation of the proposed mechanism's performance in Android malware detection across various scenarios, including general malware, adversarial attacks and zero-day malware.

#### 5.3.1. Performance comparison on general malware

The performance comparison highlights the significant superiority of the proposed mechanism over existing approaches for Android malware detection. The proposed method achieves an accuracy of 98.6% on the combination of Drebin, Androzoo and Virusshare dataset, outperforming state-of-the-art methods. Notably, it also demonstrates the lowest false positive rate (FPR) of 2.1% on Drebin. Furthermore, the execution time of 93.4 seconds is markedly lower than that of the existing mechanisms.

#### 5.3.2. Performance comparison on adversarial malware

The proposed mechanism demonstrates robust performance against various adversarial attack strategies:

- **FGSM**: 92.3% accuracy
- **JSMA**: 93.4% accuracy  
- **Salt-and-pepper**: 98.4% accuracy
- **Mimicry**: 96.5% accuracy
- **GAN**: 92.9% accuracy

#### 5.3.3. Performance on zero-day malware

For the VirusShare dataset, the proposed method achieves an accuracy of 94.1%, a high precision of 97.3%, and a recall of 90.8%, while maintaining a low false positive rate (FPR) of just 2.5%. These results indicate that the method is capable of detecting previously unseen malware samples with high reliability and minimal misclassification of benign apps.

### 5.4. Explanation of the predictions using SHAP

To interpret the classifier's predictions, we employ SHAP values (Shapley Additive Explanations). The SHAP values derived from our model align with observed malware behavior reported in the literature and public repositories [84,103–106].

Key findings include:

- **Malware families** such as Basebridge, Opfake, Gemini, and FakeInstaller often send SMS messages to premium-rate numbers, requiring permissions like SEND_SMS, RECEIVE_SMS, and WRITE_SMS [106].
- **Device information collection** through API calls like getSimSerialNumber() and permissions such as READ_PHONE_STATE are commonly used by families like Bankbot, Adrd, DroidKungfu, and Gingermaster [103].
- **Persistence mechanisms** using permissions such as RECEIVE_BOOT_COMPLETED are used by families like DroidKungfu, Golddream, Bankbot, and FakeInstaller to activate malicious behavior after device restarts.

#### Table 14: Features selected by adaptive genetic algorithm with SHAP scores

**Malware Related Features:**

| SI No | Feature | SHAP Score |
|-------|---------|------------|
| 1 | SEND_SMS | 0.239 |
| 2 | RECEIVE_SMS | 0.214 |
| 3 | READ_PHONE_STATE | 0.201 |
| 4 | READ_SMS | 0.192 |
| 5 | RECEIVE_BOOT_COMPLETED | 0.184 |
| 6 | WRITE_SMS | 0.176 |
| 7 | getSimSerialNumber | 0.168 |
| 8 | CHANGE_WIFI | 0.163 |
| 9 | ACCESS_COARSE_LOCATION | 0.150 |
| 10 | GET_TASKS | 0.142 |

**Benign Application Features:**

| SI No | Feature | SHAP Score |
|-------|---------|------------|
| 1 | FACTORY_TEST | -0.220 |
| 2 | loadClass | -0.265 |
| 3 | requestFocus | -0.213 |
| 4 | BROADCAST_STICKY | -0.175 |
| 5 | GET_ACCOUNTS | -0.164 |

The permissions and API–permission combinations identified by the proposed model are not arbitrary, but represent semantically meaningful operations that are core to the malicious intent of Android malware. Since malware, under adversarial manipulation, must preserve its core functional semantics to remain operational, the proposed detection mechanism is able to identify these persistent malicious features despite such evasive transformations.

---

## 6. Limitations and Future Work

The proposed mechanism relies on static features for detection and may be evaded by malware that employs runtime obfuscation and native code execution. Further as Android evolves, certain API's become deprecated or renamed. To overcome these limitations, future work of GA-RAM will incorporate dynamic behavioral features such as system calls [110], memory access patterns, and inter-process communication traces captured during the actual execution of Android applications. By supplementing static features with these run-time features, the mechanism can detect stealthy malware that activates only during execution and those that employs native code execution. Additionally, the integration of opcode-level features, which operate at the Dalvik bytecode level and remain stable across Android API versions, offers resilience against API-specific deprecations and renamings. Hence by combining static and dynamic features, we plan to build a robust malware detection mechanism, capable of identifying evasive and stealthy threats across diverse Android environments.

---

## 7. Conclusion

This work presented a robust and adaptive feature selection mechanism, GA-RAM, for enhancing Android malware detection. GA-RAM intelligently balances exploration and exploitation during feature evolution, enabling the identification of semantically meaningful and attack-resilient feature subsets. Extensive evaluations demonstrated the effectiveness of this approach in achieving 98.6% accuracy in detecting general malware, and maintaining over 90% detection accuracy under a variety of adversarial attack settings, including white-box, black-box, grey-box, and zero-day scenarios. The integration of SHAP explanations further enhances the model's transparency by identifying key contributing features, thus promoting interpretability and trustworthiness in predictions.

---

## References

[1-110] *References section with 110 citations following academic format...*

---

**Contact Information:**
- **Teenu S. John**: teenu.john@iiitmk.ac.in  
  Indian Institute of Information Technology and Management-Kerala, Research Centre of Cochin University of Science and Technology, Cochin, Thiruvananthapuram 695581, India

**Data Availability Statement:**
The data that support the findings of this study are available from the corresponding author T.S.John (Teenu S. John), upon reasonable request.

**Disclosure Statement:**
The authors declare that there are no relevant financial or non-financial competing interests to report.
