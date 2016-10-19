#!/usr/bin/python
# -*- coding: utf-8 -*-
import unicodedata

import gensim


class TextProcessor(object):
    """
        USAGE:

        tp = EN_TextProcessor()
        t = tp.tokenize("I really love you <3, sweethhear :)! What'ya doin'?")

        file = open("D:\\Downloads\\blog.txt", 'r')
        txt = file.read().decode('utf8')
        meta = MetaText(txt)
        meta.all_meta_attributes

        en = EN_Text()
        br = BR_Text()

        file = open("D:\\Downloads\\blog.txt", 'r')
        txt = file.read().decode('utf8')
        ldabr = br.LDA(txt, 1)[0]

        file = open("D:\\Downloads\\blog.en.txt", 'r')
        txt = file.read().decode('utf8')
        ldaen = en.LDA(txt, 1)[0]
    """

    def __init__(self, lang):
        self.LANG = lang
        self.STEMMER = LanguageTool.load(name="stemmer.{}".format(lang))
        self.S_TKR = LanguageTool.load(name="tokenizer.{}".format(lang))
        self.TGR = LanguageTool.load(name="tagger.{}".format(lang))

        self.W_TKR = [nltk.tokenize.TreebankWordTokenizer()]

    @classmethod
    def detect_language(cls, text, sample=None):
        # 01: TOKENIZE
        tokens = nltk.tokenize.TreebankWordTokenizer().tokenize(text)

        # 01.1: CLEAR TOKENS
        tokens = cls.to_lower(tokens)
        tokens = cls.remove_accents(tokens)
        tokens = cls.remove_words(tokens)
        tokens = cls.remove_punctuation(tokens)

        if not tokens:
            return None

        if sample:
            import random
            tokens_sample = []
            for i in range(0, int(sample)):
                t = random.choice(tokens)
                tokens_sample.append(t)
            tokens = tokens_sample

        # 02: CALCULATE RATIOS
        ratios = {}
        for word in tokens:
            w = Word.get(word=word.lower())
            if w is not None:
                lang = w.lang
                if lang not in ratios:
                    ratios[lang] = 0
                ratios[lang] = ratios[lang] + 1

        # 03: CHOOSE BEST RATIO
        if ratios:
            best_language = max(ratios, key=ratios.get)
            if ratios[best_language] == 0:
                return None
            else:
                return best_language
        else:
            return None

    @classmethod
    def process(cls, tokens):
        processed = cls.to_lower(tokens)
        processed = cls.remove_accents(processed)
        processed = cls.remove_words(processed)
        processed = cls.remove_punctuation(processed)
        processed = cls.remove_stopwords(processed)
        return processed

    @classmethod
    def remove_words(cls, tokens):
        words = ['http', '//', '#', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        processed = []
        for t in tokens:
            for w in words:
                found = t.find(w) != -1
                if found:
                    break
            if not found:
                processed.append(t)
        return processed

    @classmethod
    def remove_punctuation(cls, tokens):
        punctuation = [",", ".", "?", "!", ":", ';', ']', '[', ')', '(', '\'', '-', '``', "'", "''", "@", "#", "..."]
        all = []
        for t in tokens:
            # if t.isalnum():
            if t not in punctuation:
                all.append(t)
        return all

    @classmethod
    def to_lower(cls, tokens):
        all = []
        for t in tokens:
            all.append(t.lower())
        return all

    @classmethod
    def remove_accents(cls, text):
        if isinstance(text, str):
            strNorm = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore')
            return strNorm
        elif isinstance(text, list):
            all = []
            for t in text:
                all.append(TextProcessor.remove_accents(t))
            return all

    @classmethod
    def remove_stopwords(cls, tokens):
        filtered = []
        for t in tokens:
            if not TextProcessor.is_stopword(t):
                filtered.append(t)
        return filtered

    @classmethod
    def is_stopword(cls, word):
        w = Word.get(word=word)
        if w:
            return bool(w.stopword)
        else:
            return False

    def sentences(self, text):
        return self.S_TKR.tokenize(text)

    def words(self, sentence):
        if isinstance(self.W_TKR, list):  # multiple word tokenizers
            tokens = [sentence]
            for tkr in self.W_TKR:
                new_tokens = []
                for t in tokens:
                    new_tokens.extend(tkr.tokenize(t))
                tokens = new_tokens
            return tokens
        else:
            return self.W_TKR.tokenize(sentence)

    def tokenize(self, text, include_emoticon=True, remove_punctuation=False, lower_case=True):
        sents = self.sentences(text)
        tokens = []
        emoticons = []
        for sent in sents:
            rest, e = self.extract_emoticons(sent) if include_emoticon else (some, [])
            emoticons.extend(e)
            some = self.words(rest)
            cleared = self.remove_punctuation(some) if remove_punctuation else some
            lower = self.to_lower(cleared) if lower_case else cleared
            tokens.extend(lower)
        return tokens + emoticons

    def extract_emoticons(self, text):
        import re
        parts = re.split(r'[!?,\s]\s*', text)
        emoticons = []
        for p in parts:
            w = Word.get(p, lang=Languages.emoticon)
            if w and w.lang == Languages.emoticon:
                emoticons.append(p)

        for e in emoticons:
            text = text.replace(e, '')
        return text, emoticons

    def tag(self, tokens):
        return self.TGR.tag(tokens)

    def stem(self, tokens):
        l = []
        for t in tokens:
            if t:
                s = self.STEMMER.stem(t)
                l.append(s)
        return l

    def is_subjective(self, stemmed):
        if stemmed:
            # 01: dictionary classification
            subjective_amt = 0
            for w in stemmed:
                if Word.get(word=w):
                    subjective_amt = subjective_amt + 1

            subj_ratio = float(subjective_amt) / len(stemmed)
            if subj_ratio > 0.0:
                return True

            # 02: ensemble classification
            if self.CL_SUBJ is not None:
                return self.CL_SUBJ.classify({'stemmed': stemmed})
            else:
                return False
        else:
            return False

    def find_sentiment(self, stemmed, subjective=None):
        neutral = None
        if stemmed:
            # 01: dictionary classification
            happy_amt = 0
            sad_amt = 0
            for word in stemmed:
                w = Word.get(word=word)
                p = w.emotion == Word.EMOTION_POS
                s = bool(w.subjective)
                if subjective is None:
                    ok = p
                elif subjective == True:
                    ok = p and s
                else:
                    ok = p and not s
                if ok is not None:
                    if ok == True:
                        happy_amt = happy_amt + 1
                    else:
                        sad_amt = sad_amt + 1

            if happy_amt > 0 or sad_amt > 0:
                winner = happy_amt > sad_amt if happy_amt != sad_amt else neutral
                return winner

            # 02: ensemble classification
            classifier = self.CL_SUBJ_SENT if subjective else self.CL_OBJ_SENT
            if classifier is not None:
                return classifier.classify({'stemmed': stemmed})
        return neutral

    def find_gender(self, stemmed):
        undefined = None
        if stemmed:
            # 01: ensemble classification
            classifier = self.CL_GENDER
            if classifier is not None:
                return classifier.classify({'stemmed': stemmed})
        return undefined

    def _filter_latent(self, topics):
        pos_filters = ['NPROP', 'ART', 'KC', 'PREP',
                       'AT', 'CC', 'TO', 'DT', 'IN']

        # FILTERS
        filtered = []
        for topic in topics:
            f = []
            for freq, w in topic:
                if not self.is_stopword(w.lower()):
                    pos = self.tag([w])[0][1]
                    if pos not in pos_filters:
                        f.append((freq, w))
            filtered.append(f)
        return filtered

    def LDA(self, text, nTopics=1):
        if not isinstance(text, list):
            text = self.tokenize(text)
        tokens = self.remove_punctuation(text)

        dictionary = gensim.corpora.Dictionary([tokens])
        corpus = dictionary.doc2bow(tokens)

        lda = gensim.models.ldamodel.LdaModel(corpus=[corpus], id2word=dictionary, num_topics=nTopics, update_every=1,
                                              chunksize=10000, passes=1)

        topics = lda.show_topics(num_topics=1, num_words=50, log=False, formatted=False)

        return self._filter_latent(topics=topics)

    def LSI(self, text, nTopics=1):
        tokens = self.tokenize(text)

        dictionary = gensim.corpora.Dictionary([tokens])
        corpus = dictionary.doc2bow(tokens)

        lsi = gensim.models.lsimodel.LsiModel(corpus=corpus, id2word=dictionary, num_topics=nTopics, chunksize=10000)
        topics = lsi.show_topics(num_topics=1, num_words=50, log=False, formatted=False)

        return self._filter_latent(topics=topics)


class BrTextProcessor(TextProcessor):
    def __init__(self):
        super().__init__(lang= 'br')


class EnTextProcessor(TextProcessor):
    def __init__(self):
        super().__init__(lang='en')