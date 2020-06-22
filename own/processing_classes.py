import re
from nltk.corpus import wordnet
import enchant
from nltk.metrics import edit_distance

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()



class RegexReplacer(object):
	def __init__(self, replacement_patterns):
		self.patterns = replacement_patterns # list of tuples

	def replace(self, text):
		cur_text = text
		for (pattern, repl) in self.patterns:
			cur_replace_regex = re.compile(pattern, re.IGNORECASE)
			cur_text = cur_replace_regex.sub(repl, cur_text)
		return cur_text
			

class SpellingReplacer(object):
	def __init__(self, dict_name="en", max_dist=2):
		self.spell_dict = enchant.Dict(dict_name)
		self.max_dist = max_dist
	
	def replace(self, word, print_b = False):
		regex_not = re.compile(r".*_not") # matching changed words like did_not

		if self.spell_dict.check(word):
			return word
		else:
			
			if bool(regex_not.match(word)): # return changed words without suggestions
				return word
			else:
				suggestions = self.spell_dict.suggest(word)

				if suggestions and edit_distance(word, suggestions[0])<= self.max_dist:
					if print_b:
						print(word," -> ", suggestions[0])
					return suggestions[0]
				else:
					return word
	
	def replace_list(self, word_list, print_b = False):
		return [self.replace(word, print_b).lower() for word in word_list]

class LemmaStemmer(object):
	def __init__(self):
		pass

	def perform(self, word_list, print_b = False):
		stemmed_words = []
		for word in word_list:
			if (len(wordnet.synsets(word))!= 0):
				stemmed_words.append(lemmatizer.lemmatize(word, pos = wordnet.synsets(word)[0].pos()))
			else:
				stemmed_words.append(stemmer.stem(word))
		if print_b:
			print(stemmed_words)
		return stemmed_words


		 


class AntonymReplacer(object):
	def replace(self, word, pos=None):
		antonyms = set()
		for syn in wordnet.synsets(word, pos=pos):
			for lemma in syn.lemmas():
				for antonym in lemma.antonyms():
					antonyms.add(antonym.name())
		if len(antonyms) == 0:
			return None
		else:
			return list(antonyms)[0]
			

	def replace_negations(self, sent, print_b = False):
		i, l = 0, len(sent)
		words = []
		while i < l:
			word = sent[i]
			if word == 'not' and i+1 < l: # only match real "not" words
				ant = self.replace(sent[i+1])
				if print_b:
					print(word, " -> ", ant)
				if ant:
					words.append(ant)
					i += 2
					continue
			words.append(word)
			i += 1
		return words

	def replace_lists(self, word_lists, print_b = False):
		replaced_word_lists = []	
		for word_list in word_lists:
			# print(word_list)
			replaced_word_lists.append(self.replace_negations(word_list, print_b))

		return replaced_word_lists
