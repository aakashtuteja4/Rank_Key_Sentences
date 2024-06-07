import networkx as nx
import numpy as np
 
from nltk.tokenize.punkt import PunktSentenceTokenizer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

def textrank(document):
    sentence_tokenizer = PunktSentenceTokenizer()
    sentences = sentence_tokenizer.tokenize(document)
 
    bow_matrix = CountVectorizer().fit_transform(sentences)
    normalized = TfidfTransformer().fit_transform(bow_matrix)
 
    similarity_graph = normalized * normalized.T
 
    nx_graph = nx.from_scipy_sparse_array(similarity_graph)
    scores = nx.pagerank(nx_graph)
    return sorted(((scores[i],s.replace('\n', ' ')) for i,s in enumerate(sentences)),
                  reverse=True)


document = """To Sherlock Holmes she is always the woman. I have
seldom heard him mention her under any other name. In his eyes she
eclipses and predominates the whole of her sex. It was not that he
felt any emotion akin to love for Irene Adler. All emotions, and that
one particularly, were abhorrent to his cold, precise but admirably
balanced mind. He was, I take it, the most perfect reasoning and
observing machine that the world has seen, but as a lover he would
have placed himself in a false position. He never spoke of the softer
passions, save with a gibe and a sneer. They were admirable things for
the observer-excellent for drawing the veil from men’s motives and
actions. But for the trained reasoner to admit such intrusions into
his own delicate and finely adjusted temperament was to introduce a
distracting factor which might throw a doubt upon all his mental
results. Grit in a sensitive instrument, or a crack in one of his own
high-power lenses, would not be more disturbing than a strong emotion
in a nature such as his. And yet there was but one woman to him, and
that woman was the late Irene Adler, of dubious and questionable
memory."""


document = """
Children living in Japan’s hottest city will be given specially designed umbrellas to protect them from the heat, 
after a summer that saw record-breaking temperatures in many parts of the country.Local authorities in Kumagaya in 
Saitama prefecture have devised an umbrella that keeps out the rain and doubles as a parasol, the Mainichi Shimbun reported. 
The umbrellas, which bear the city’s logo and weigh just 336 grams, will be distributed to 9,000 primary schoolchildren next 
week, the newspaper said. Kumagaya, a city of about 195,000 located 60km north of Tokyo, regularly records the highest temperatures 
in Japan partly as a result of warm downslope winds created by the Foehn Effect. The city’s government has for the past two years 
advised younger children to shield themselves from the sun with regular umbrellas on their way to and from school to prevent 
heatstroke, but some questioned their ability to block out sunlight. Alarmed by a rise in the number of days when the mercury 
rises to at least 35C, the city decided to hand out the yellow fibreglass umbrellas, including to children who live in Kumagaya 
but attend schools outside the city, the Mainichi said. The heat-busting brollies will also force children to maintain a 
reasonable distance from each other, eliminating the need for them to wear masks to prevent the spread of the coronavirus, 
it added. The measure has come a little late in the day, however. Japan battled its worst heatwave since records began in 
1875 in late June, after a premature end to the rainy season. The city of Isesaki, north of Tokyo, registered the country’s 
highest-ever temperature for that month, at 40.2C, beating the previous June record of 39.8C set in 2011. Tokyo experienced 
several consecutive days of 35C-plus heat, prompting the government to warn people to save energy or face power cuts, while 
Kumagaya and five other locations marked highs above 40C on 1 July. Kumagaya’s reputation for furnace-like temperatures 
was sealed in July 2018, when it battled an all-time high temperature of 41.1C – an unenviable record it shares with 
the city of Hamamatsu in central Japan. On Friday, the maximum temperature for Kumagaya was a far more comfortable 
26C, according to the meteorological agency, although it forecast a rerun to the low 30s next week. Officials had 
hoped to distribute the umbrellas before the school summer holidays began were delayed by the Covid-19 pandemic. 
Global heating has prompted Japan’s government to take extra measures and issue a slew of advice on how to prevent 
heatstroke. Almost all classrooms in public primary and middle schools now have air conditioners, according to the 
Asahi Shimbun, while the education ministry last year urged teachers to instruct children to wear cool clothing and 
hats, and to keep hydrated when they travel to and from school. The pandemic has frustrated attempts to keep children 
cool at school, however, with teachers reporting that many are reluctant to remove their masks, even with 
encouragement from staff.
"""
for i in textrank(document):
    print(i)