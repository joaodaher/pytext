#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
.. module:: text_analysis
   :platform: OSX

.. moduleauthor:: Joao Daher <joao.daher@cti.gov.br>
"""
import re
import math
import numpy
import nltk
import gensim

from bi.text_models import Word, LanguageTool, Emotions, Languages

from text.processor import BR_TextProcessor, EN_TextProcessor, TextProcessor


class MetaText(object):
    HANDLERS = {Languages.brazilian: BR_TextProcessor(),
                Languages.english: EN_TextProcessor(),
                Languages.undefined: EN_TextProcessor(),
                Languages.emoticon: EN_TextProcessor()}

    def calculate(self):
        for w in self.WORDS:
            self.STEMMED.extend(tp.stem(w))
            self.TAGGED.extend(tp.tag(w))

        self.C = len(self.RAW)
        self.LETTERS = MetaText._count_char(self.RAW, "^[\w_-]*$")
        self.UPPER = MetaText._count_char(self.RAW, "^[A-Z_-]*$")
        self.NUMBERS = MetaText._count_char(self.RAW, "^[\d]*$")
        self.WHITE = MetaText._count_char(self.RAW, "^[ ]*$")
        self.TAB = MetaText._count_char(self.RAW, "^[\t]*$")
        
        self.N = len(self.WORDS)

        self.SIZES = []
        self.FREQ = {}
        for w in self.WORDS:            
            self.SIZES.append(len(w))

        self.FREQ = dict(nltk.FreqDist(self.WORDS))

        self.V = dict(nltk.FreqDist(self.FREQ.values())) 

        self.HXLEGO = []
        self.HXDISLEGO = []
        for w, t in self.FREQ.viewitems():
            if t == 1:
                self.HXLEGO.append(w)
            elif t == 2:
                self.HXDISLEGO.append(w)

        self.S = len(self.SENTENCES)

    def __init__(self, text, nMaxLengthFreq=20):
        # Control parameters
        self.nMaxLengthFreq = nMaxLengthFreq #Tamanho maximo de palavra a ser considerado na frequencia do tamanho de palavras       

        text = re.sub(r'http[s]?://.*\s', '', text)
        self.RAW = TextProcessor.remove_accents(text)

        self.lang = TextProcessor.detect_language(self.RAW, sample=100)
        tp = self.HANDLERS[self.lang]

        self.STEMMED = []
        self.TAGGED = []

        self.PARAGRAPHS = []
        self.SENTENCES = []
        self.WORDS = []

        #delimiters = '\n','. \n', '! \n', '?\n', '.\n', '!\n', '?\n', '... \n' #, '... \n'#, ' \n ' #, " .\n", " !\n", ' ?\n'
        #regexPattern = '|'.join(map(re.escape, delimiters))
        #for paragraph in re.split(regexPattern, self.RAW):
        for paragraph in self.RAW.split('\n'):       
            p = []
            for sentence in tp.sentences(paragraph): 
                words = tp.tokenize(text=sentence, remove_punctuation=True)
                
                self.WORDS.extend(words)
                self.SENTENCES.append(sentence)

                p.append(words)
            self.PARAGRAPHS.append(p)

        self.STEMMED = []
        self.TAGGED = []

        self.C = None
        self.LETTERS = None
        self.UPPER = None
        self.NUMBERS = None
        self.WHITE = None
        self.TAB = None
        
        self.N = len(self.WORDS)

        self.SIZES = []
        self.FREQ = {}

        self.V = {}

        self.HXLEGO = []
        self.HXDISLEGO = []
        
        self.S = None
    
    def __iadd__(self, mt):
        if not isinstance(mt, MetaText):
            raise Exception(u"Unable to add object to {} class".format(mt.__class__.__name__))

        self.RAW = "{}\n{}".format(self.RAW, mt.RAW)

        sample_size = 0.01 * (self.N + mt.N)
        self.lang = TextProcessor.detect_language(self.RAW , sample=sample_size if sample_size <1000 else 1000)
        tp = self.HANDLERS[self.lang]

        self.STEMMED.extend(mt.STEMMED)
        self.TAGGED.extend(mt.TAGGED)
        
        self.PARAGRAPHS.extend(mt.PARAGRAPHS)
        self.SENTENCES.extend(mt.SENTENCES)
        self.WORDS.extend(mt.WORDS)

        if self.C is not None or mt.C is not None:
            #force calculation
            if self.C is not None: self.calculate()
            if mt.C is not None: mt.calculate()

            self.C += mt.C
            self.LETTERS += mt.LETTERS
            self.UPPER += mt.UPPER
            self.NUMBERS += mt.NUMBERS
            self.WHITE += mt.WHITE
            self.TAB += mt.TAB
        
            self.N += mt.N

            self.SIZES.extend(mt.SIZES)

            for w, f in mt.FREQ.viewitems():
                try: self.FREQ[w] += f
                except IndexError: self.FREQ[w] = f

            for w, f in mt.V.viewitems():
                try: self.V[w] += f
                except IndexError: self.V[w] = f

            self.HXLEGO.extend(mt.HXLEGO)
            self.HXDISLEGO.extend(mt.HXDISLEGO)

            self.S += mt.S

        return self


    '''
    -----------------------------------------------------------------------------------------------------------------------
    CHARACTERS (C)
    -----------------------------------------------------------------------------------------------------------------------
    '''

    @property
    def C01(self):
        return self.C

    @property
    def C02(self):
        return self.LETTERS / float(self.C)

    @property
    def C03(self):
        return self.UPPER / float(self.C)

    @property
    def C04(self):
        return self.NUMBERS / float(self.C)

    @property
    def C05(self):
        return self.WHITE / float(self.C)

    @property
    def C06(self):
        return self.TAB / float(self.C)
        
    @property
    def C07(self):
        return self._count_char(self.RAW, r'\'') / float(self.C)

    @property
    def C08(self):
        return self._count_char(self.RAW, r'\,') / float(self.C)

    @property
    def C09(self):
        return self._count_char(self.RAW, r'\:') / float(self.C)

    @property
    def C10(self):
        return self._count_char(self.RAW, r'\;') / float(self.C)

    @property
    def C11(self):
        return self._count_char(self.RAW, r'\?') / float(self.C)

    @property
    def C12(self):
        return self._count_char(self.RAW, r'\?\?+') / float(self.C)

    @property
    def C13(self):
        return self._count_char(self.RAW, r'\!') / float(self.C)

    @property
    def C14(self):
        return self._count_char(self.RAW, r'\!\!+') / float(self.C)
    
    @property
    def C15(self):
        return self._count_char(self.RAW, r'\.') / float(self.C)
    
    @property
    def C16(self):
        return self._count_char(self.RAW, r'\.\.+') / float(self.C)     
        
    '''
    -----------------------------------------------------------------------------------------------------------------------
    WORDS (W)
    -----------------------------------------------------------------------------------------------------------------------
    '''
    @property
    def W01(self):  
        return self.N

    @property
    def W02(self):     
        n = []
        for w in self.WORDS:
            n.append(len(w))
        return float(numpy.mean(n))

    @property
    def W03(self):
        if self.N == 0:
            return 0
        return len(self.FREQ) / float(self.N)

    @property
    def W04(self):    
        t = 0
        for s in self.SIZES:
            if s >= 6:
                t = t + 1
        return t

    @property
    def W05(self):
        t = 0
        for s in self.SIZES:
            if s <= 3:
                t = t + 1
        return t

    @property
    def W06(self):
        if self.N == 0:
            return 0

        return len(self.HXLEGO) / float(self.N)

    @property
    def W07(self):
        if self.N == 0:
            return 0
        return len(self.HXDISLEGO) / float(self.N)
        

    @property
    def W08(self):
        if self.N == 0:
            return 0

        sum = 0
        for f, amt in self.V.viewitems():
            it = amt * math.pow(f/float(self.N), 2)
            sum = sum + it
        yule = (-1.0/self.N + sum)
        return yule

    @property
    def W09(self):
        if self.N <= 1:
            return 0

        simpson = 0
        i=1
        #OBS: Sentenças de apenas uma palavra tornam a funcao indefinida        
        for f, amt in self.V.viewitems():
            it = amt *(i / float(self.N)) * ((i-1.0) / (self.N - 1.0))
            simpson = simpson + it
            i=i+1
       
        return simpson

    @property
    def W10(self):
        if self.N == 0:
            return 0

        return len(self.HXLEGO) / float(len(self.V))

    @property
    def W11(self):
        if self.N == 0:
            return 0

        hlego_count = len(self.HXLEGO)
        v_count = len(self.FREQ)
        if hlego_count == v_count:
            return 0
        else:
            return (100.0 * math.log10(self.N)) / float(1.0-(hlego_count/float(v_count))) 
    
    @property
    def W12(self):
        if self.N == 0:
            return 0

        entropy = 0
        for f, amt in self.V.viewitems():
            it = amt * (-math.log10(f/float(self.N))) * (f/float(self.N))
            entropy = entropy + it
        #max = float(math.log10(self.N))
        #return entropy
        return entropy

    @property
    def W13(self):
        if self.N == 0:
            return 0
               
        vFreq = numpy.zeros([self.nMaxLengthFreq])      
        for item in nltk.FreqDist(self.SIZES).items():           
            if item[0] < self.nMaxLengthFreq:
                vFreq[item[0]-1] = item[1]/float(self.N)
        return list(vFreq)
        
    '''
    -----------------------------------------------------------------------------------------------------------------------
    TEXT STRUCTURE (TS)
    -----------------------------------------------------------------------------------------------------------------------
    ''' 
    @property
    def TS01(self):
        #Total number of sentences (S)        
        return self.S

    @property
    def TS02(self):
        #Total number of paragraphs
        return len(self.PARAGRAPHS)

    @property
    def TS03(self):
        #Average number of sentences per paragraph
        sents_per_paragraph = []
        for p in self.PARAGRAPHS:
            sents_per_paragraph.append(len(p))
        return numpy.average(sents_per_paragraph)
        
    @property
    def TS04(self):
        #Average number of words per paragraph
        words_per_paragraph = []
        for p in self.PARAGRAPHS:
            total_words = 0
            for s in p:
                for w in s:
                    if w.isalpha():
                        total_words = total_words + 1
            words_per_paragraph.append(total_words)
        return numpy.average(words_per_paragraph)

    @property
    def TS05(self):
        #Average number of characters per paragraph
        chars_per_paragraph = []
        for p in self.PARAGRAPHS:
            total_chars = 0
            for s in p:
                for w in s:
                    total_chars = total_chars + len(w)
            chars_per_paragraph.append(total_chars)

        return numpy.average(chars_per_paragraph)

    @property
    def TS06(self):
        #Average number of words per sentence
        words_per_sentence = []
        for p in self.PARAGRAPHS:
            for s in p:
                total_words = 0
                for w in s:
                    if w.isalpha():
                        total_words = total_words + 1
                words_per_sentence.append(total_words)
        return numpy.average(words_per_sentence)

    @property
    def TS07(self):
        #Number of sentences beginning with upper case/S
        amt = 0        
        for p in self.PARAGRAPHS:            
            for s in p:
                if len(s)>0:
                    first_char = s[0][0]
                    if first_char.isupper():
                        amt = amt + 1                
        return amt / float(self.S)       

    @property
    def TS08(self):
        #Number of sentences beginning with lower case/S
        amt = 0
        for p in self.PARAGRAPHS:
            for s in p:
                if len(s)>0:
                    first_char = s[0][0]
                    if first_char.islower():
                        amt = amt + 1
        return amt / float(self.S)

    @property
    def TS09(self):
        #Number of blank lines/total number of paragraohhs
        blank = 0
        for p in self.PARAGRAPHS:
            if len(p) == 0:
                blank = blank + 1
        return blank / float(self.TS02)

    @property
    def TS10(self):
        #Average length of non-blank line
        lenghts = []
        for p in self.PARAGRAPHS:
            lenght = 0
            for s in p:                
                for w in s:
                    #if len(w)>0:                        
                    lenght = lenght + len(w)
            lenghts.append(lenght)
        return numpy.average(lenghts)
        
        
    '''
    -----------------------------------------------------------------------------------------------------------------------
    TEXT MORFOLOGY (TM)
    -----------------------------------------------------------------------------------------------------------------------
    '''
    @property
    def TM01(self):
        articles = []
        for word, tag in self.TAGGED:
            if tag == 'ART':
                articles.append(word)
        freqdist = nltk.FreqDist(articles)

        article_ratio = {}
        for article, freq in freqdist.viewitems():
            article_ratio[article] = freq / float(self.N)
        return article_ratio

    @property
    def TM02(self):
        freqdist = self._morfo_freq(['PROADJ', 'PRO-KS', 'PROPESS', 'PRO-KS-REL', 'PRO-SUB'])

        pronoun_ratio = {}
        for pronoun, freq in freqdist.viewitems():
            pronoun_ratio[pronoun] = freq / float(self.N)
        return pronoun_ratio

    @property
    def TM03(self):
        freqdist = self._morfo_freq('VAUX')

        verb_ratio = {}
        for verb, freq in freqdist.viewitems():
            verb_ratio[verb] = freq / float(self.N)
        return verb_ratio

    @property
    def TM04(self):
        freqdist = self._morfo_freq(['KC', 'KS'])

        conj_ratio = {}
        for conj, freq in freqdist.viewitems():
            conj_ratio[conj] = freq / float(self.N)
        return conj_ratio

    @property
    def TM05(self):
        freqdist = self._morfo_freq('IN')

        inter_ratio = {}
        for inter, freq in freqdist.viewitems():
            inter_ratio[inter] = freq / float(self.N)
        return inter_ratio

    @property
    def TM06(self):
        freqdist = self._morfo_freq('PREP')

        prep_ratio = {}
        for prep, freq in freqdist.viewitems():
            prep_ratio[prep] = freq / float(self.N)
        return prep_ratio
     
    '''
    -----------------------------------------------------------------------------------------------------------------------
    SEMANTIC BASED DICTIONARY (SBD)
    -----------------------------------------------------------------------------------------------------------------------
    '''
    @property    
    def SBD01(self):
        if self.N == 0:
            return 0

        n = 0
        for word in self.WORDS:
            w = Word.get(word=word)
            if w and w.subjective is True and emotion == Emotions.negative:
                n =+ 1
        return n / float(self.N)

    @property    
    def SBD02(self):
        if self.N == 0:
            return 0

        n = 0
        for word in self.WORDS:
            w = Word.get(word=word)
            if w and w.subjective is True and emotion == Emotions.positive:
                n =+ 1
        return n / float(self.N)

    @property    
    def SBD03(self):
        if self.N == 0:
            return 0

        n = 0
        for word in self.WORDS:
            w = Word.get(word=word)
            if w and w.subjective is False and emotion == Emotions.negative:
                n =+ 1
        return n / float(self.N)

    @property    
    def SBD04(self):
        return 0.0

    '''
    -----------------------------------------------------------------------------------------------------------------------
    EMOTICONS (E)
    -----------------------------------------------------------------------------------------------------------------------
    '''
    @property
    def E01(self):
        n = 0
        for word in self.WORDS:
            w = Word.get(word=word)
            if w and w.lang == Languages.emoticon:
                n =+ 1
        return n

    @property
    def E02(self):
        n = 0
        for word in self.WORDS:
            w = Word.get(word=word)
            if w and w.lang == Languages.emoticon and w.emotion == Emotions.positive:
                n =+ 1
        return n

    @property
    def E03(self):
        n = 0
        for word in self.WORDS:
            w = Word.get(word=word)
            if w and w.lang == Languages.emoticon and w.emotion == Emotions.negative:
                n =+ 1
        return n

    @property
    def E04(self):
        n = 0
        for word in self.WORDS:
            w = Word.get(word=word)
            if w and w.lang == Languages.emoticon and w.emotion == Emotions.neutral:
                n =+ 1
        return n


    '''
    ------------------------------------------------------------------------------------------
    UTILS
    ------------------------------------------------------------------------------------------
    '''
    def _morfo_freq(self, part_of_speech):
        if isinstance(part_of_speech, str):
            part_of_speech = [part_of_speech]

        words = []
        for word, tag in self.TAGGED:
            if tag in part_of_speech:
                words.append(word)
        return nltk.FreqDist(words)

    @staticmethod
    def _count_char(text, regex = None):        
        if regex is None:
            return len(text)
        else:
            t = 0
            for c in text:
                if re.match(regex, c):
                    t = t + 1
            #return len(re.findall(regex, text))
            return t

    @property
    def all_meta_attributes(self):
        names = [('C', 16), ('W', 13), ('TS', 10), ('E', 4), ('SBD', 4)]
        vMA = numpy.array([])
        lMA = []

        for name, n in names:
            for i in range(0, n):
                att_name = "{0}{1:02d}".format(name, i+1)
                att = getattr(self, att_name)

                if isinstance(att, numpy.ndarray.__class__):
                    for item in att:
                        vMA = numpy.append(vMA, item)
                if isinstance(att, dict):
                    for item in att.values():
                        vMA = numpy.append(vMA, item)
                else:
                    c = att.__class__
                    vMA = numpy.append(vMA, att)
                lMA.append((att_name, vMA[-1]))

        return vMA
