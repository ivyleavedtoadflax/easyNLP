import nltk

s = 'Albert Einstein was born on MArch 14, 1879 in Ulm, Germany'

tags = nltk.pos_tag(s.split())

print(tags)

print(nltk.ne_chunk(tags))

nltk.ne_chunk(tags).draw()
