ngramlength = 5
windowsize = 1

file = "a,b,c,d,e\nb,c,d,e,f\nc,d,e,f,g"
ngrams = map(lambda line: line.split(","), file.split("\n"))

map = {}
for ngram in ngrams:
   map[" ".join(ngram[0:ngramlength-windowsize])] = ngram[ngramlength-windowsize]

sentence = ngrams[0]
while True:
  key = " ".join(sentence[-ngramlength+windowsize:])
  if key not in map:
    break
  sentence.append(map[key])
print(" ".join(sentence))
