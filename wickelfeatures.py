#%% wickelfeatures
"""
Replication of the past tense learning model by Rumelhart and McClelland (1986)

Original model description:
words are represented as sets wickelphones (contiguous trigrams of characters) with two extra characters
indicating beginning and the end of the word. Importantly the set of wickelphones is unordered.
e.g. camel => (#ca, cam, ame, mel, el#) // order of items can be altered

It's assumed that no word has 2 identical triplets, so order can be inferred from the set of triplets.
(there are few exceptions of this rule actually: banana, femininity, ukelele, rococo, balalaika)

Phonemes are elements of the phonetic alphabet (similar, but not identical to IPA of English).
For simplicity in this implementation letters from the spelling of the word are converted into
closest phonemes in the phonetic alphabet to obtain the phonological representation of the word.

There are 10 features (combined into 4 dimensions) that identify a phoneme. Each phoneme can have
only one value on each of the dimensions. Thus, each phoneme is identified by presence of
4 features out of 10.

Here is the dimensions/features:
1.  ['interrupted', 'continuous_consonant', 'vowel']
2.  values in this dimension have different interpretations depending on the values of the first dimension:
    for 'interrupted' the values ['stop', 'nasal'],
    for 'continuous_consonant' - ['fricative', 'liquid'],
    for 'vowel' - ['high', 'low']
3.  place of articulation = ['front', 'middle', 'back']
4.  values interpretations depend on the first dimension:
    for ['interrupted', 'continuous_consonant'] its ['voiced', 'unvoiced']
    for 'vowels' its ['long', 'short']

Wickelfeature_detector - is a combination of 3 features. E.g. ('interrupted','frontal','interrupted').
Generally speaking, there are 10x10x10 possible detectors. Each detector is used to flag the presence of
its particular combination of 3 consecutive features in a given wickelphone. For the detector
to be activated the first feature should be found in the 1st phoneme, 2nd feature in the 2nd phoneme,
and the 3rd feature in the 3rd phoneme.
E.g. if the wickelphone is 'k','e','m', then the detector ('interrupted','frontal','continuous_consonant')
will not be activated (as 3rd phoneme 'm' has no 'continuous_consonant' feature), whereas the detector
('stop','vowel','interrupted') will be activated as all 3 of its features are present in corresponding
phonemes in the wickelphone ('k' - 'stop', 'i' - 'vowel', 'm' - 'interrupted'). Each detector responds
to multiple wickelphones

In practice not all the detectors are used in the model (as they are redundant). Only those detectors
are used that have their 1st feature and the 3rd feature belonging to the same dimension.
('interrupted','front','vowel') - is used, whereas ('interrupted', 'front', 'voiced') is not.
Thus, there 260 detectors satisfying 'same dimension for features 1 and 3' property.
Additionally there is a boundary character (#) and its phoneme_code ('0000000000') are added to
the inventory of phonemes and detectors such as ('boundary', feature1, feature2) and
(feature1, feature2, boundary) are used for all combinations of feature1 and feature2. I.e. there are
1x10x10 - 'beginning' boundary detector (for each 10x10 combination of feature1 and 2) and 10x10x1
'ending' boundary detectors. So, the total number of detectors is 260+100+100=460
    Each wickelphone activates exactly 16 detectors. E.g. non-boundary wickelphone - any dimension from
    phoneme 1 and 3 (4 dimensions) combines with any dimension from the phoneme 2 (4 dimensions), 4X4=16
    For boundary wickelphone there are 16 detectors as well (1 for phoneme 1 (boundary sign) combines
    with 4 dimensions from phoneme 2 and 4 dimensions from phoneme 3 (1X4X4=16).

========================================================================================================
Terminology in this implementation:
1.  feature - 'interrupted','continuous_consonant',...., 'unvoiced/short' - names/interpretations
    of the real language phonetic features that identify the phonemes.
2.  feature_key - combination of dimension indices and codes within each dimension
    that allows to identify the feature e.g. (100,1) is the key for 'interrupted',
    (100,10,2) - the key for 'stop' (dimension 1 code /100/ is necessary as the name of the feature varies
    for different dimensions)
3.  feature_code - 1-hot 10-dimensional vector encoding a given feature.
    It's used either as a string or as a list.
4.  phoneme_code - 4-hot 10-dimensional vector encoding a given phoneme. It's used either as a string or as a list or
    as a combination of list/strings.
    e.g. ['100', '10', '100', '10'], '1001010010', [1,0,0,1,0,1,0,0,1,0] - are all different formats for the phoneme 'b'

5.  wickelphone - triplet (trigram) of characters that the word is split into.
6.  wickelfeature_detector - a triplet of features. ('1000000000','0100000000','0000000001') - tuple of 3 feature_codes
7.  wickelphone_activation_pattern (= wickelfeature representation of a wickelphone) - 16-hot 460-dimensional vector
    (the 16 wickelfeature_detectors activated by a given wickelphone have value =1, the rest = 0).
8.  word_activation_pattern (= wickelfeature representation of a word) - x-hot 460-dimensional vector
    (all wickelfeature_detectors corresponding to any of
    the wickelphones in which the word can be broken into are activated).
========================================================================================================

BINDING NETWORK
Binding network is a decoding network which converts from the wickelfeature representation to either the
wickelphone representation or phonological representation format.
input units : wickelfeature_detectors, output units: wickelphones.
The difficulty in decoding wickelfeature representation of a word back into a phonological representation
caused by the fact that a) any particular wickelfeature is part of the encoding of multiple unrelated
wickelphones, b) there is a significant element of noise in the wickelfeature representation

"""

from boilerplate import *  # import settings and modules
from itertools import product


# descriptions of feature coding keys (medium_feature_code => feature_name)
feature_keys = {
                ('100', 1): 'interrupted',
                ('010', 1): 'continuous_consonant',
                ('001', 1): 'vowel',
                ('100', '10', 2): 'stop',
                ('100', '01', 2): 'nasal',
                ('010', '10', 2): 'fricative',
                ('010', '01', 2): 'liquid',
                ('001', '10', 2): 'high',
                ('001', '01', 2): 'low',
                ('100', 3): 'front',
                ('010', 3): 'middle',
                ('001', 3): 'back',
                ('100', '10', 4): 'voiced',
                ('100', '01', 4): 'unvoiced',
                ('010', '10', 4): 'voiced',
                ('010', '01', 4): 'unvoiced',
                ('001', '10', 4): 'long',
                ('001', '01', 4): 'short'
                }  # code descriptions


# activation patterns of all phonemes
phoneme_codes = {
                # interrupted, stop
                'b': ['100', '10', '100', '10'],
                'p': ['100', '10', '100', '01'],
                'd': ['100', '10', '010', '10'],
                't': ['100', '10', '010', '01'],
                'g': ['100', '10', '001', '10'],
                'k': ['100', '10', '001', '01'],

                # interrupted, nasal
                'm': ['100', '01', '100', '10'],
                'n': ['100', '01', '010', '10'],
                'ng': ['100', '01', '001', '10'],

                # continuous consonant, fricative
                'v-Dh': ['010', '10', '100', '10'],
                'f-Th': ['010', '10', '100', '01'],
                'z': ['010', '10', '010', '10'],
                's': ['010', '10', '010', '01'],
                'Z-j': ['010', '10', '001', '10'],
                'S-C': ['010', '10', '001', '01'],

                # continuous consonant, liquid
                'w-l': ['010', '01', '100', '10'],
                'r': ['010', '01', '010', '10'],
                'y': ['010', '01', '001', '10'],
                'h': ['010', '01', '001', '01'],

                # vowels, high
                'E': ['001', '10', '100', '10'],
                'i': ['001', '10', '100', '01'],
                'O': ['001', '10', '010', '10'],
                'U': ['001', '10', '001', '10'],
                'u': ['001', '10', '001', '01'],

                # vowels, low
                'A': ['001', '01', '100', '10'],
                'e': ['001', '01', '100', '01'],
                'I': ['001', '01', '010', '10'],
                'a': ['001', '01', '010', '01'],
                'W': ['001', '01', '001', '10'],
                'o': ['001', '01', '001', '01'],

                # boundary
                '#': ['000', '00', '000', '00']
                }  # codes of all phonemes


# convert phoneme code (represented as 10 items 4-hot string) to the phoneme
def phoneme_from_code(phoneme_code_string):
    """
    converts code of the phoneme (activation pattern of the features) into phoneme description
    also returns phoneme itself
    :param phoneme_code_string:
    :return:
    """
    try:
        d1 = phoneme_code_string[0:3]
        d2 = phoneme_code_string[3:5]
        d3 = phoneme_code_string[5:8]
        d4 = phoneme_code_string[8:10]

        d1d = feature_keys[d1, 1]
        d2d = feature_keys[d1, d2, 2]
        d3d = feature_keys[d3, 3]
        d4d = feature_keys[d1, d4, 4]

        code_to_phoneme = {tuple(phoneme_codes[k]): k for k in phoneme_codes}
        phoneme = code_to_phoneme[d1, d2, d3, d4]

    except KeyError:
        return 'None'

    return phoneme, [d1d, d2d, d3d, d4d]

    # test
    activation_pattern = '1001010010'
    phoneme_from_code(activation_pattern)
    phoneme_from_code('0100110010')


# returns dictionary of feature names corresponding to feature_codes (represented as 1-hot 10-char string)
def get_feature_code_dictionary():
    """
    some features have different interpretations (names, labels) depending on values of other features.
    e.g. feature_code '00010000' interpreted as 'stop' for 'interrupted' phonemes, and at the same time
    it's interpreted as 'high' for vowels.
    in this case the function returne the dictionary value that contains both names ['stop','high']
    :return: dictionary of {feature_code_string: [feature_name,...]
    """
    e_keys = list(feature_keys.keys())
    codes_to_features = {}
    for k in e_keys:
        if k[-1] == 1:
            pt = k[0]+'0'*7
        elif k[-1] == 2:
            pt = '0'*3 + k[1] + '0'*5
        elif k[-1] == 3:
            pt = '0'*5 + k[0] + '0'*2
        else:
            pt = '0'*8 + k[1]
        codes_to_features[pt] = codes_to_features.get(pt, []) + [feature_keys[k]]
    codes_to_features['0000000000'] = ['boundary']  # add code for the bounary character
    return codes_to_features

    # test
    fc = get_feature_code_dictionary()
    len(fc)  # 11


# get the description (list of feature names) of the phoneme
def phoneme_description(phoneme):
    d1, d2, d3, d4 = phoneme_codes[phoneme]
    d1d = feature_keys[d1, 1]
    d2d = feature_keys[d1, d2, 2]
    d3d = feature_keys[d3, 3]
    d4d = feature_keys[d1, d4, 4]
    return [d1d, d2d, d3d, d4d]

    # test
    phoneme_description('I')
    phoneme_description('E')


# generates all legitimate triplets of activation patterns used as wickelfeatures detectors in the original model
def get_all_wickelfeature_detectors():
    """
    wickelfeature detector - a unit that activates when the wickelphone (trigram) has a combination of features
    corresponding to a given detector. e.g. detector_1 looks for (f1,f2,f3) combination. where f1 = 'interrupted'
    f2 = 'vowel', f3='back'. if first phoneme of the wickelphone has f1, 2nd has f2, and 3rd has f3, then
    detector_1 is activated.

    naive wickeflfeatures detectors combine all possible values of all features of the phoneme (on all dimensions)
    there are 11 x 11 x 11 naive wickelfeature detectors. This approach produces too many redundant wickelfeatures.

    optimized wickelfeature detectors combine all values of the central (2nd) phoneme with combinations of features
    of the same dimension of the preceding (1st) and following (3rd) phonemes. e.g. there is a detector for
    ('interrupted', 'front', 'continuous_consonant') - since 'interrupted' and 'continuous_consonant' are values
    of the same dimension, but there are no detectors for ('interrupted', 'front', 'high') as 'interrupted' and
    'high' are on different dimensions.

    there are 9 combinations of features of the 1st and 3rd phonemes on the 1st dimension,
    4 on the 2nd dimension, 9 on the 3rd dimension and 4 on the 4th dimension. - total 26 combinations.
    each of the 26 combinations can be combined with any of 10 features of the central (2nd) phoneme.
    so in optimized version there 26 x 10 = 260 wickelfeature detectors.

    this function produces and returns their codes

    :return: list of 3 item tuples each representing combination of features for a given wickelfeature detector
    """

    wz = '0'*9  # template wickelfeature

    wickel_features = [wz[:n] + '1' + wz[n:] for n in range(10)]  # 10 possible codes of wickelfeatures
    wf1 = wickel_features[0:3]  # codes representing dimension 1
    wf2 = wickel_features[3:5]  # codes representing dimension 2
    wf3 = wickel_features[5:8]  # dimension 3
    wf4 = wickel_features[8:10]  # dimension 4

    wfx = list(product(wf1, wickel_features, wf1)) + list(product(wf2, wickel_features, wf2))
    wfx += list(product(wf3, wickel_features, wf3)) + list(product(wf4, wickel_features, wf4))

    wf_b = [(wz+'0', x[0], x[1]) for x in product(wickel_features, wickel_features)]
    wf_e = [(x[0], x[1], wz+'0') for x in product(wickel_features, wickel_features)]

    return wfx + wf_b + wf_e

    # test
    wfx = get_all_wickelfeature_detectors()  # full list of wickelfeatures
    len(wfx)  # 460 as expected. 260 + 100 + 100
    wf_example = wfx[100]  # ('01000000000', '01000000000', '10000000000')
    fc = get_feature_code_dictionary()
    wfx_description = [[fc[e] for e in wf_e] for wf_e in wfx]
    len(wfx_description)
    pp(wfx_description)
    wfx_description[459]
    wfx_description[200]


# returns the list of wickelfeature detectors corresponding to a given wickelphone
def get_detectors_for_wickelphone(wickelphone, blur_ratio=0):
    """
    :param wickelphone: triplet (list) of phonemes
    :param blur_ratio: probability of activation of the blurring features
    :return: list of wickelfeatures corresponding to the wickelphone
    """

    def code_to_vector(code, dim):  # convert feature_key into feature_code
        if dim == 1:
            return code + '0'*7
        elif dim == 2:
            return '0'*3 + code + '0'*5
        elif dim == 3:
            return '0'*5 + code + '0'*2
        else:
            return '0'*8 + code


    # initialize feature code lists
    features_1_3_combo = []
    features_1 = []
    features_2 = []
    features_3 = []

    wf_code = [phoneme_codes[p] for p in wickelphone]  # convert phonemes into phoneme_codes

    # convert feature_keys into feature_codes
    for dim in range(4):
        features_1.append(code_to_vector(wf_code[0][dim], dim + 1))  # features_codes of the 1st phoneme
        features_2.append(code_to_vector(wf_code[1][dim], dim + 1))  # feature_codes of the 2nd phoneme
        features_3.append(code_to_vector(wf_code[2][dim], dim + 1))  # feature_codes of the 3rd phoneme
        # tuples of feature_codes of the same dimensions of the 1st and 3rd phoneme.
        features_1_3_combo.append((code_to_vector(wf_code[0][dim], dim+1), code_to_vector(wf_code[2][dim], dim+1)))

    # generate wickelfeature detectors >>
    if not wickelphone[0] == '#' and not wickelphone[2] == '#':
        # any feature of the 2nd phoneme and pair of features of 1st and 3rd phonemes of the same dimension
        wfs = [(x[0][0], x[1], x[0][1]) for x in product(features_1_3_combo, features_2)]
    elif wickelphone[0] == '#':
        # left boundary character + any combination of features of phonemes 2 and 3
        wfs = [('0000000000', x[0], x[1]) for x in product(features_2, features_3)]
    else:
        # any combination of features of phonemes 1 and 2 + right boundary character
        wfs = [(x[0], x[1], '0000000000') for x in product(features_1, features_2)]

    # blurring
    if blur_ratio > 0:

        features = list(get_feature_code_dictionary().keys())
        features_1_2_combo = list(product(features_1, features_2))  # list of tuples of f1, f2 combinations
        features_2_3_combo = list(product(features_2, features_3))  # list of tuples of f2, f3 combinations

        blur_1_2 = list((x[0][0], x[0][1], x[1]) for x in product(features_1_2_combo, features))
        blur_2_3 = list((x[0], x[1][0], x[1][1]) for x in product(features, features_2_3_combo))

        num_extra_detectors = int(blur_ratio * len(blur_1_2))
        extra_detectors_1 = sample(blur_1_2, num_extra_detectors)
        extra_detectors_2 = sample(blur_2_3, num_extra_detectors)

        wfs = wfs + extra_detectors_1 + extra_detectors_2

    return wfs

    wickelphone = ['b', 'r', 'd']
    blur_ratio = 0
    wfs = get_detectors_for_wickelphone(wickelphone, blur_ratio=0.5)
    wfs = get_detectors_for_wickelphone(['#', 'k', 'o'], blur_ratio=0.5)
    len(wfs)
    pp(wfs)


# adjust the word spelling
def word_spelling(word):
    spelling_keys = {
                    'v': ['v-Dh'],
                    'f': ['f-Th'],
                    'l': ['w-l'],
                    'j': ['Z-j'],
                    'c': ['k'],
                    'w': ['W'],
                    'x': ['k', 's'],
                    'q': ['k'],
                    'é': ['e']
                    }
    word = list(word)
    word = [spelling_keys[c] if c in spelling_keys else c for c in word]
    word = [c for e in word for c in e]
    return word

    # test:
    word_spelling('vox')


# convert a word into an activation pattern of wickelfeature detectors
def word_activation_pattern(word, i_return=False, blur_ratio=0.0):
    """
    split the word into wickelphones, compute activation pattern for each wickelphone, union wickelphone
    activation to get the word activation pattern
    >> the word activation pattern is activation of all wickelfeature detectors
    corresponding to each of wickelphones in a word.
    e.g. (simplified and reduced dimensions for clarity) if 'come' is split into 'com' and 'ome' and 'com' activation
    pattern is '10001' and 'ome' activation pattern '01001', then the whole word activation pattern is '11001'
    :param word: string, natural language word
    :param blur_ratio: how much blurring to add
    :param i_return: if True returns indices of wickelfeature detectors, otherwise returns x-hot list
    :return: x-hot vector of length=460 (encoding of the word in wickelfeature detectors)
    """
    word = '#' + word + '#'  # add boundary features
    word = word_spelling(word)  # adjust spelling and convert into a list
    word_in_triplets = [word[i:i+3] for i in range(len(word)-2)]  # break down to triplets
    wickel_features_detectors = get_all_wickelfeature_detectors()  # load all wickelfeature_detectors(460)
    activation = ['0'] * len(wickel_features_detectors)  # initialize activation

    for t in word_in_triplets:  # loop over each individual triple (wickelphone)
        # 16 active wickelfeature detectors corresponding to wickelphone + blurring detectors
        wickelphone_activation = get_detectors_for_wickelphone(t, blur_ratio)
        activation_wf = ['0' if wf not in wickelphone_activation else '1' for wf in wickel_features_detectors]
        activation = ['1' if x == '1' or y == '1' else '0' for x, y in zip(activation, activation_wf)]

    if i_return:  # convert into list of indices
        activation = [i for i in range(len(activation)) if activation[i] == '1']

    return activation

    # test:
    word = 'i'  # there is no wickelfeature detector for 1 letter words, so this returns []
    word = 'paradox'
    x = word_activation_pattern(word, i_return=False, blur_ratio=0.0)
    x = word_activation_pattern(word, i_return=True, blur_ratio=0.0)
    len(x)  # each wickelphone activates 16 detectors. A word contains len(word) wickelphones.
    x
    # the maximum len(x) = 16 * len(word), but some wickelphones may activate the same detectors.
    1 - len(x) / (len(word) * (16 + int(176*0.9)))  # repetition ratio


# check whether wickelfeature encoding produces unique activation patterns for all words (yes)
def test_activation_uniqueness(n_words=100, blur_ratio=0.0):
    """
    The function collects activation patterns of words from the dictionary and checks their uniqueness
    the capacity of the wickelfeature detector is very large (2**460), but it's not immediately obvious
    that the encoding method produces unique activation patterns for every word. So, this function
    checks if this is true or not. It turns out it is true.
    :return: list of tuples of words with identical activation
    """

    frequency_file = 'words/word_frequency_df_ready.csv'
    df = pd.read_csv(frequency_file, sep='\t')
    lexicon = list(df.w1)
    lexicon = [str(lx) for lx in lexicon]
    lexicon = [lx.lower() for lx in lexicon if not ("'" in lx or "-" in lx or '/' in lx)]
    lexicon = list(set(lexicon))  # len of lexicon is 69957

    activations = []
    for word in lexicon[:n_words]:
        try:
            activations.append(word_activation_pattern(word, blur_ratio))
        except KeyError:
            print(word)
            activations.append(['-1']*460)

    word_duplicates = []

    for i in range(n_words):
        for j in range(i+1, n_words):
            if activations[i] == activations[j]:
                word_duplicates.append((lexicon[i], lexicon[j]))

    return word_duplicates

    # test:
    x = test_activation_uniqueness(10000, blur_ratio=0.5)
    # the duplicates 'c' and 'k' as in simplified spelling 'c' translates to 'k'
    # with one-letter words like i or a as they have 2 boundaries characters and there are no detectors for that
    len(x)
    x


# collect distribution of word activation patterns by size (number of active wickelfeature detectors)
def test_representation_size(n_words=100, blur_ratio=0.0):
    """
    compute distribution of activation pattern length (active detectors)
    :return:
    """

    frequency_file = 'words/word_frequency_df_ready.csv'
    df = pd.read_csv(frequency_file, sep='\t')
    lexicon = list(df.w1)
    lexicon = [str(lx) for lx in lexicon]
    lexicon = [lx.lower() for lx in lexicon if not ("'" in lx or "-" in lx or '/' in lx)]
    lexicon = list(set(lexicon))  # len of lexicon is 69957

    pattern_size = np.zeros((2, n_words))
    for i, w in enumerate(lexicon[:n_words]):
        pattern_size[0, i] = len(w)
        wap = word_activation_pattern(w, i_return=False, blur_ratio=blur_ratio)
        wap = [int(x) for x in wap]
        pattern_size[1, i] = sum(wap)

    df_pattern_size = pd.DataFrame(columns=['word_length', 'pattern_size'], data=pattern_size.T)
    df_pattern_size['detector_per_letter'] = df_pattern_size.pattern_size/df_pattern_size.word_length
    return df_pattern_size

    # test:
    x = test_representation_size(n_words=5000, blur_ratio=0.9)
    x
    plt.scatter(x=x.word_length, y=x.pattern_size)
    plt.show()
    px.box(x, x='word_length', y='detector_per_letter',
           title=str(n_words) + ' words. blur_ratio=' + str(blur_ratio)).show()




































