#!/usr/bin/env python
"""
Train and predict using a Hidden Markov Model part-of-speech tagger.

Usage:
  hw5.py training_file test_file
"""

import optparse
import collections
import math

import hw5_common

# Smoothing methods
NO_SMOOTHING = 'None'  # Return 0 for the probability of unseen events
ADD_ONE_SMOOTHING = 'AddOne'  # Add a count of 1 for every possible event.
# *** Add additional smoothing methods here ***

# Unknown word handling methods
PREDICT_ZERO = 'None'  # Return 0 for the probability of unseen words

# If p is the most common part of speech in the training data,
# Pr(unknown word | p) = 1; Pr(unknown word | <anything else>) = 0
PREDICT_MOST_COMMON_PART_OF_SPEECH = 'MostCommonPos'
# *** Add additional unknown-word-handling methods here ***

PREDICT_CUSTOM = "Custom"


class BaselineModel:
  '''A baseline part-of-speech tagger.

  Fields:
    dictionary: map from a word to the most common part-of-speech for that word.
    default: the most common overall part of speech.
  '''
  def __init__(self, training_data):
    '''Train a baseline most-common-part-of-speech classifier.

    Args:
      training_data: a list of pos, word pairs:
    '''
    words = set(word for pos, word in training_data)
    poss = set(pos for pos, word in training_data)
    total = {}
    for pos in poss:
      total[pos] = 0
    gather = {}
    for word in words:
      gather[word] = {}
      for pos in poss:
        gather[word][pos] = 0
    for (part, word) in training_data:
      total[part] += 1
      gather[word][part] += 1

    self.dictionary = {}
    for word in gather.keys():
      val = 0
      pos = ""
      for part in gather[word].keys():
        if gather[word][part] > val:
          val = gather[word][part]
          pos = part
      self.dictionary[word] = pos

    val = 0
    pos = ""
    for part in total.keys():
      if total[part] > val:
        val = total[part]
        pos = part
    self.default = pos
        

  def predict_sentence(self, sentence):
    return [self.dictionary.get(word, self.default) for word in sentence]


class HiddenMarkovModel:
  def __init__(self, order, emission, transition, parts_of_speech, known):
    # Order 0 -> unigram model, order 1 -> bigram, order 2 -> trigram, etc.
    self.order = order
    # Emission probabilities, a map from (pos, word) to Pr(word|pos)
    self.emission = emission
    # Transition probabilities
    # For a bigram model, a map from (pos0, pos1) to Pr(pos1|pos0)
    self.transition = transition
    # A set of parts of speech known by the model
    self.parts_of_speech = parts_of_speech
    # A set of words known by the model
    self.known_words = known
    self.unknown = PREDICT_ZERO

  def predict_sentence(self, sentence):
    return self.find_best_path(self.compute_lattice(sentence))

  def compute_bigramlattice(self, sentence):
    if sentence == []:
      return None
    lattice = [{} for i in range(len(sentence)+2)]
    for i in range(len(lattice)-2):
      for pos in self.parts_of_speech:
        lattice[i+1][pos] = (-float('inf'), None)
    lattice[0] = {'<s>': (0, None)}
    for i in range(len(sentence)):
      word = sentence[i]
      if self.unknown == PREDICT_ZERO and not word in self.known_words:
        prev = ''
        val = -float('inf')
        for pos in lattice[i].keys():
          if lattice[i][pos][0] >= val:
            prev = pos
            val = lattice[i][pos][0]
        lattice.append({'':(-float('inf'), prev)})
        continue
      if word.lower() in self.known_words:
        word = word.lower()
      for pos in self.parts_of_speech:
        if pos == '<s>':
          continue
        prob = 0
        if not word in self.known_words:
          if self.unknown == PREDICT_CUSTOM:
            if "ly" in word[len(word)-2:]:
              if pos == 'J':
                prob = -5
              elif pos != 'R':
                prob = -float('inf')
            elif 'able' in word[len(word)-4:]:
              if pos == 'J':
                prob = -1
              elif pos != self.mostcommon:
                prob = -float('inf')
            elif 'al' in word[len(word)-2:] or 'ious' in word[len(word)-4:] or 'ous' in word[len(word)-3:] or 'ish' in word[len(word)-3:]:
              if pos == 'J':
                prob = -0.1
              elif pos != self.mostcommon:
                prob = -float('inf')
            elif 'esque' in word[len(word)-5:] or 'ical' in word[len(word)-4:] or 'y' in word[len(word)-1:]:
              if pos != 'J' and pos != self.mostcommon:
                prob = -float('inf')
            elif 'ful' in word[len(word)-3:] or 'ic' in word[len(word)-2:] or 'ive' in word[len(word)-3:] or 'less' in word[len(word)-4:]:
              if pos != 'J':
                prob = -float('inf')
            elif 'ed' in word[len(word)-2:] or 'ing' in word[len(word)-3:] or 'ize' in word[len(word)-3:]:
              if pos != 'V':
                prob = -float('inf')
            elif 'dis' in word[:3] or 'un' in word[:2] or 'up' in word[:2] or 'with' in word[:4]:
              if pos != 'V' and pos != self.mostcommon:
                prob = -float('inf')
            elif 'er' in word[len(word)-2:]:
              if pos == 'J':
                prob = -2
              elif pos != self.mostcommon:
                prob = -float('inf')
            elif 'en' in word[len(word)-2:] or 'ate' in word[len(word)-3:] or 're' in word[:2] or 'ify' in word[len(word)-3:] or 'be' in word[:2] or 'de' in word[:2] or 'en' in word[:2] or 'ise' in word[len(word)-3:] or 's' in word[len(word)-1:] or 'a' in word[:1]:
              if pos == 'V':
                prob = -0.5
              elif pos != self.mostcommon:
                prob = -float('inf')
            elif filter(str.isdigit, word) != '':
              if pos != 'M':
                prob = -float('inf')
            else:
              if pos != self.mostcommon:
                prob = -float('inf')
          else:
            if pos != 'F' and pos != self.mostcommon:
              prob = -float('inf')
        else:
          prob = self.emission[(pos, word)]
        prev = ''
        val = -float('inf')
        for checkpos in lattice[i].keys():
          check = prob
          nextprob = self.transition.get((checkpos, pos),-float('inf'))
          check += lattice[i][checkpos][0] + nextprob
          if check >= val:
            val = check
            prev = checkpos
        lattice[i+1][pos] = (val, prev)
    prev = ''
    val = -float('inf')
    for pos in lattice[len(lattice)-2].keys():
      nextprob = -float('inf')
      if (pos, '<s>') in self.transition:
        nextprob = self.transition[(pos, '<s>')]
      check = lattice[len(lattice)-2][pos][0] + nextprob
      if check >= val:
        val = check
        prev = pos
    lattice[len(lattice)-1] = {"<s>":(val, prev)}
    return lattice

  def compute_trigramlattice(self, sentence):
    lattice = [{} for i in range(len(sentence)+4)]
    lattice[0] = {(None,'<s0>'):(0, None)}
    lattice[1] = {('<s0>','<s1>'):(0, None)}
    for i in range(len(lattice)-4):
      for pos0 in self.parts_of_speech:
        if pos0 == '<s1>' or pos0 == '<s0>':
          continue
        for pos1 in self.parts_of_speech:
          if pos1 == '<s1>' or pos1 == '<s0>':
            continue
          lattice[i+2][pos0,pos1] = (-float('inf'),None)
    for i in range(len(sentence)):
      word = sentence[i]
      poss_parts = self.parts_of_speech
      if not word in self.known_words:
        if self.unknown == PREDICT_MOST_COMMON_PART_OF_SPEECH:
          poss_parts = [self.mostcommon]
      for pos2 in self.parts_of_speech:
        if pos2 == '<s0>' or pos2 == '<s1>':
          continue
        prob = self.emission.get(pos2, 0)
        for (_,pos1) in lattice[i+1].keys():
          val = -float('inf')
          prev = ''
          for (pos0,_) in lattice[i+1].keys():
            check = prob + lattice[i+1][(pos0, pos1)][0] + self.transition[(pos0,pos1),pos2]
            if check >= val:
              val = check
              prev = pos0
          lattice[i+2][pos1,pos2] = (val, prev)
    for (_,pos1) in lattice[len(lattice)-3].keys():
      val = -float('inf')
      prev = ''
      for (pos0,_) in lattice[len(lattice)-3].keys():
        check = lattice[len(lattice)-3][(pos0, pos1)][0] + self.transition[(pos0, pos1),'<s0>']
        if check >= val:
          val = check
          prev = pos0
      lattice[len(lattice)-2][pos1,'<s0>'] = (val, prev)
    val = -float('inf')
    prev = ''
    for (pos,_) in lattice[len(lattice)-2].keys():
      check = lattice[len(lattice)-2][pos,'<s0>'][0] + self.transition[(pos0,'<s0>'),'<s1>']
      if check >= val:
        val = check
        prev = pos
    lattice[len(lattice)-1]['<s0>','<s1>'] = (check,prev)
    return lattice

  def compute_lattice(self, sentence):
    """Compute the Viterbi lattice for an example sentence.

    Args:
      sentence: a list of words, not including the <s> tokens on either end.

    Returns:
      FOR ORDER 1 Markov models:
      lattice: [{pos: (score, prev_pos)}]
        That is, lattice[i][pos] = (score, prev_pos) where score is the
        log probability of the most likely pos/word sequence ending in word i
        having part-of-speech pos, and prev_pos is the part-of-speech of word i-1
        in that sequence.

        i=0 is the <s> token before the sentence
        i=1 is the first word of the sentence.
        len(lattice) = len(sentence) + 2.

      FOR ORDER 2 Markov models: ??? (extra credit)
    """
    if self.order == 1:
      return self.compute_bigramlattice(sentence)
    elif self.order == 2:
      return self.compute_trigramlattice(sentence)

  @staticmethod
  def traintrigram(training_data, smoothing, unknown_handling, order):
    words = set(word for pos, word in training_data)
    poss = set(pos for pos, word in training_data)
    poss.remove('<s>')
    poss = poss.union(set(['<s1>','<s0>']))
    emit = {}
    transit = {}
    tcount = {}
    for word in words:
      emit[word] = {}
      for pos in poss:
        emit[word][pos] = 0
    for pos0 in poss:
      for pos1 in poss:
        tcount[pos0,pos1] = 0
        transit[pos0,pos1] = {}
        for pos2 in poss:
          transit[pos0, pos1][pos2] = 0
    wcount = {"C":0,"D":0,"E":0,"F":0,"G":0,"I":0,"J":0,"L":0,"M":0,"N":0,"P":0,"R":0,"T":0,"U":0,"V":0,"X":0,"Y":0,"Z":0,"<s0>":0,"<s1>":0}
    prevpos1 = ''
    prevpos2 = ''
    for i in range(len(training_data)):
      pos = training_data[i][0]
      word = training_data[i][1]
      if not pos == '<s>':
        emit[word][pos] += 1
        wcount[pos] += 1
      if i == 0 or (i==1 and pos=='<s1>'):
        if pos == '<s>':
          prevpos1 = '<s0>'
          prevpos2 = '<s1>'
        else:
          prevpos1 = prevpos2
          prevpos2 = pos
        continue
      if pos == '<s>':
        pos = '<s0>'
      transit[(prevpos1, prevpos2)][pos] += 1
      tcount[(prevpos1, prevpos2)] += 1
      if pos == '<s0>':
        prevpos1 = '<s0>'
        prevpos2 = '<s1>'
      else:
        prevpos1 = prevpos2
        prevpos2 = pos

    val = max(wcount.values())
    mostcommon = wcount.keys()[wcount.values().index(val)]
    emission = {}
    for pos in poss:
      if pos == '<s0>' or pos == '<s1>':
        continue
      for word in words:
        total = emit[word][pos]
        count = wcount[pos]
        if total == 0:
          emission[(pos, word)] = -float('inf')
        else:
          emission[(pos, word)] = math.log(total / float(count))
    emission['<s0>','<s0>'] = 0
    emission['<s1>','<s1>'] = 0
    transition = {}
    for pos1 in poss:
      for pos2 in poss:
        if pos2 == '<s1>' and pos1 != '<s0>':
          continue
        for pos3 in poss:
          if pos3 == '<s1>' and pos2 != '<s0>':
            continue
          if pos3 == '<s1>' and pos2 == '<s0>':
            transition[(pos1, pos2),pos3] = 0
          else:
            total = transit[pos1, pos2][pos3]
            count = tcount[(pos1, pos2)]
            if total == 0:
              transition[(pos1, pos2),pos3] = -float('inf')
            else:
              transition[(pos1, pos2),pos3] = math.log(total / float(count))
    h = HiddenMarkovModel(2, emission, transition, poss, words)
    h.unknown = unknown_handling
    h.mostcommon = mostcommon
    return h
    
  @staticmethod
  def trainbigram(training_data, smoothing, unknown_handling, order):
    words = set(word for pos, word in training_data)
    poss = set(pos for pos, word in training_data)
    emit = {}
    for word in words:
      emit[word] = {}
      for pos in poss:
        emit[word][pos] = 0
        
    transit = {}
    wordcount = {}
    poscount = {}
    for pos0 in poss:
      transit[pos0] = {}
      wordcount[pos0] = 0
      poscount[pos0] = 0
      for pos1 in poss:
        transit[pos0][pos1] = 0

    total = 0
    prevpos = ''
    for i in range(len(training_data)):
      pos1 = training_data[i][0]
      word = training_data[i][1]
      wordcount[pos1] += 1
      emit[word][pos1] += 1
      if i == 0:
        prevpos = pos1
        continue
      pos0 = prevpos
      poscount[pos0] += 1
      transit[pos0][pos1] += 1
      prevpos = pos1
    mostcommon = ''
    val = 0
    for pos in wordcount.keys():
      if wordcount[pos] > val:
        mostcommon = pos
        val = wordcount[pos]
    emission = {}
    transition = {}
    for pos0 in poss:
      for word in words:
        count = wordcount[pos0]
        total = emit[word][pos0]
        if smoothing == ADD_ONE_SMOOTHING:
          total += 1
          count += len(words)
        if total == 0:
          emission[(pos0, word)] = -float('inf')
        else:
          emission[(pos0, word)] = math.log(total / float(count))
      for pos1 in poss:
        total = transit[pos0][pos1]
        count = poscount[pos0]
        if smoothing == ADD_ONE_SMOOTHING:
          total += 1
          count += len(poss)
        if total == 0:
          transition[(pos0, pos1)] = -float('inf')
        else:
          transition[(pos0, pos1)] = math.log(total / float(count))
    h = HiddenMarkovModel(order, emission, transition, poss, words)
    h.unknown = unknown_handling
    h.mostcommon = mostcommon
    return h
  
  @staticmethod
  def train(training_data,
      smoothing=NO_SMOOTHING,
      unknown_handling=PREDICT_CUSTOM,
      order=1):
      # You can add additional keyword parameters here if you wish.
    '''Train a hidden-Markov-model part-of-speech tagger.

    Args:
      training_data: A list of pairs of a word and a part-of-speech.
      smoothing: The method to use for smoothing probabilities.
         Must be one of the _SMOOTHING constants above.
      unknown_handling: The method to use for handling unknown words.
         Must be one of the PREDICT_ constants above.
      order: The Markov order; the number of previous parts of speech to
        condition on in the transition probabilities.  A bigram model is order 1.

    Returns:
      A HiddenMarkovModel instance.
    '''
    if order == 1:
      return HiddenMarkovModel.trainbigram(training_data, smoothing, unknown_handling, order)
    elif order == 2:
      return HiddenMarkovModel.traintrigram(training_data, smoothing, unknown_handling, order)
    

  @staticmethod
  def find_best_path(lattice):
    """Return the best path backwards through a complete Viterbi lattice.

    Args:
      FOR ORDER 1 MARKOV MODELS (bigram):
        lattice: [{pos: (score, prev_pos)}].  See compute_lattice for details.

    Returns:
      FOR ORDER 1 MARKOV MODELS (bigram):
        A list of parts of speech.  Does not include the <s> tokens surrounding
        the sentence, so the length of the return value is 2 less than the length
        of the lattice.
    """
    if lattice == None:
      return ['<s>', '<s>']
    path = []
    if lattice[0].keys()[0] == '<s>':
      for i in reversed(range(len(lattice))):
        prev = '<s>'
        if len(path) != 0:
          prev = path[len(path)-1]
        nextpath = lattice[i][prev][1]
        if nextpath == '<s>':
          break
        path.append(nextpath)
    else:
      for i in reversed(range(len(lattice))):
        prev1 = '<s1>'
        prev2 = '<s0>'
        if len(path) == 1:
          prev2 = path[0]
          prev1 = '<s0>'
        elif len(path) > 1:
          prev2 = path[len(path)-1]
          prev1 = path[len(path)-2]
        nextpath = lattice[i][prev2,prev1][1]
        if nextpath == '<s1>':
          break
        path.append(nextpath)
    path.reverse()
    return path


def main():
  parser = optparse.OptionParser()
  parser.add_option('-s', '--smoothing', choices=(NO_SMOOTHING,
    ADD_ONE_SMOOTHING), default=NO_SMOOTHING)
  parser.add_option('-o', '--order', default=1, type=int)
  parser.add_option('-u', '--unknown',
      choices=(PREDICT_ZERO, PREDICT_MOST_COMMON_PART_OF_SPEECH,),
      default=PREDICT_ZERO)
  options, args = parser.parse_args()
  train_filename, test_filename = args
  training_data = hw5_common.read_part_of_speech_file(train_filename)
  if options.order == 0:
    model = BaselineModel(training_data)
  else:
    model = HiddenMarkovModel.train(
        training_data, options.smoothing, options.unknown, options.order)
  predictions = hw5_common.get_predictions(
      test_filename, model.predict_sentence)
  for word, prediction, true_pos in predictions:
    print word, prediction, true_pos

if __name__ == '__main__':
  main()
