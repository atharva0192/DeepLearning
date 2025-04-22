---
Author: Atharva Chavan
Date: 02-03-2025
Sources: CVBL and Coursera
Duration:
---
```dataview
TABLE WITHOUT ID 
	link(file.name,file.aliases[0]) as "Note",
	Comments as "Description"
WHERE 
	contains(file.path, this.file.folder) 
	AND file.name != this.file.name
SORT file.aliases[0] ASC
```

## Introduction


### Deep Learning and Machine Learning
- Machine Learning algorithms work on the human defined representations and input features
- ML becomes just optimising weights to make best final prediction
- ##### What is Deep Learning?
	- Learning using multiple neural network
	- A ML subfield of learning representations of data. Exceptionally effective at learning patterns
	- Deep Learning algorithms try to learn the representation by using a heirarchy of multiple layers
	- With tons of information it starts to understand the data and respond in useful ways


