import json
from urllib.request import urlopen
from bs4 import BeautifulSoup
from mathMLtoOP import *
from tempGraphing import *
from nltk.tokenize import sent_tokenize

# ----------------------------------- Debugging Subtree Similarity -----------------------------------
# Description: Custom In Order Traversal of Tree to parse through all Children nodes in order 
# @Param root = root of Tree
# ----------------------------------------------------------------------------------------------------
def IOT(root):
    if root is None:
        return
    if root.children:
        # Traverse the leftmost subtree
        IOT(root.children[0])
        try:
            # Attempt to print the root.value, encoding and decoding as necessary
            print(root.value.encode('utf-8').decode('utf-8', 'ignore'))
        except UnicodeEncodeError:
            # Handle the UnicodeEncodeError gracefully
            print("Unable to print root.value due to encoding issue")
        # Traverse the next subtree, if it exists
        for i in range(1, len(root.children)):
            # Print the root value again
            try:
                # Attempt to print the root.value, encoding and decoding as necessary
                print(root.value.encode('utf-8').decode('utf-8', 'ignore'))
            except UnicodeEncodeError:
                # Handle the UnicodeEncodeError gracefully
                print("Unable to print root.value due to encoding issue")

            # Traverse the next subtree
            IOT(root.children[i])
    else:
        # If there are no children, just print the root value
        try:
            # Attempt to print the root.value, encoding and decoding as necessary
            print(root.value.encode('utf-8').decode('utf-8', 'ignore'))
        except UnicodeEncodeError:
            # Handle the UnicodeEncodeError gracefully
            print("Unable to print root.value due to encoding issue")

# ----------------------------------- preProcessing -----------------------------------
# Description: Processing initial HTML with flags for convenient identification 
# @Param url = url of Mathematical Document
# -------------------------------------------------------------------------------------
def cleanUp(url):
    # Set up Beautiful Soup Parser
    html = urlopen(url).read()
    soup = BeautifulSoup(html, 'html.parser')

    # Label block equations with "mathequation"
    for script in soup("math", {"display" : "block"}):
        script.insert_before("mathequation")        # All block equations have unique string prior
    
    # Replace MathML with the text "unicodeError"
    for script in soup(['math']):
        script.string = "unicodeError"              # All block equations have unique string prior 

    # Get rid of annoying citations
    for script in soup(['cite']):
        script.extract()            # Removed

    # Adding paragraph break markers (parabreak) before each paragraph
    for script in soup(['p']):                      # For all the tags that have 'p'
        if script.get('class') == ['ltx_p']:        # If class tag is labelled with 'ltx_p'
            script.insert_before("parabreak")       # Insert marker before each paragraph

    equationlink_marker = "equationlink"
    # Adding edge markers (equationlink) before each equation link
    for script in soup.find_all(['a']):  # For all the tags that are anchor links
        if script.get('class') == ['ltx_ref']:  # If class tag is labelled with 'ltx_ref'
            script.insert_before(equationlink_marker)  # Insert marker before each equation link

    # Get final processed text
    text = soup.get_text(' ', strip=True)           # Strip whitespace from the beginning and end of each bit of text; No more '\n' in text

    # Remove References OR Acknowledgments (Last) section
    text = text.split("References")     # Split string at "References"
    text = text[0]                      # Take only string before "References"
    text = text.split("Acknowledgments")
    text = text[0]

    return text

# ----------------------------- Block Equation Extraction -----------------------------
# Description: Extracts all Block/Numbered Equations from a Mathematical Text
# @Param url = url of Mathematical Document
# -------------------------------------------------------------------------------------
def eqExtract(url):
    html = urlopen(url).read()
    soup = BeautifulSoup(html, 'html.parser')
    ret = []

    # Find all td tags with rowspan= any int
    td_tags_with_rowspan_one = soup.find_all('td', {'rowspan': True})

    for td_tag in td_tags_with_rowspan_one:
        # Find the closest ancestor that is either a table or tbody tag
        ancestor_table_or_tbody = td_tag.find_parent(['table', 'tbody'])

        while ancestor_table_or_tbody:
            # Create a new element with the insert text
            marker = soup.new_tag("span", text='mathmarker', **{'class': 'mathmarker'})

            if ancestor_table_or_tbody.get('id'):
                ancestor_table_or_tbody.insert_before(marker)
                ret.append(ancestor_table_or_tbody)
                break
            else:
                # If id not found, go to the next ancestor
                ancestor_table_or_tbody = ancestor_table_or_tbody.find_parent(['table', 'body'])
    
    # Replace MathML with the text "unicodeError"
    for script in soup(['math']):
        script.string = "unicodeError"              # All block equations have unique string prior 

    # Get rid of annoying citations
    for script in soup(['cite']):
        script.extract()            # Removed

    # Adding paragraph break markers (parabreak) before each paragraph
    for script in soup(['p']):                      # For all the tags that have 'p'
        if script.get('class') == ['ltx_p']:        # If class tag is labelled with 'ltx_p'
            script.insert_before("parabreak")       # Insert marker before each paragraph

    # Adding edge markers (edge) before each equation
    for script in soup(['a']):                          # For all the tags that have 'a'
        if script.get('class') == ['ltx_ref']:          # If class tag is labelled with 'ltx_ref'
            script.insert_before("equationlink")        # Insert marker before each equation
        
    # Check for elements with class "mathmarker" and skip processing them
    for script in soup.find_all(recursive=True):
        if script.get('class') == ['mathmarker']:
            script.insert_before("mathmarker")

    # Get final processed text (including markers)
    text = soup.get_text(' ', strip=True)  # Get text with some whitespace

    # Remove References OR Acknowledgments (Last) section
    text = (text.rsplit("References", 1))[0]
    text = text.split("Acknowledgments")[0]  # Split string at "Acknowledgments" and take only string before it
    return ret, text


'''
    # Find all table MathML elements
    mathML_all = soup.find_all("table")

    # Loop through all tables
    for table in mathML_all:
        # Check if the table has a td element with rowspan=any int
        td_with_rowspan = table.find('td', {'rowspan': True})
    
        if td_with_rowspan:
            ret.append(table)
'''

# ----------------------------- Block Equation id Extraction -----------------------------
# Description: Extracts all Block/Numbered Equation ID's from a Mathematical Text
# @Param mathML = list of mathML equation elements
# -------------------------------------------------------------------------------------
def idExtract(mathML):
    math_ids = []
    for tag in mathML:
        if tag.get('id'):
            math_ids.append(tag.get('id'))

    return math_ids

# ----------------------------------- Array of Strings/Words -----------------------------------
# Description: Converting entire text to an array of strings/words
# @Param text = Original text of HTML Document
# ----------------------------------------------------------------------------------------------
def arrOfStrings(text):
    output = []
    temp = ''
    # Converting to array of strings
    for i in range(len(text)):
        temp += (text[i])                   # Adding chars together until find a space
        if text[i] == ' ':                  # Once space is found,
            output.append(temp[:-1])        # Add string to output array 
            temp = ''
            continue
    # print(len(output))
    return output

# ----------------------------------- # Words per Sentence -----------------------------------
# Description: Keeps track of # of words in each sentence; Use for para interval extension
# @Param text = Original text of HTML Document
# --------------------------------------------------------------------------------------------
def sentenceCount(text):
    # Split String on Sentences
    tokenized = sent_tokenize(text)
    # Debugging for printing entire text w/o references AND split into sentences
    # print('Text Split into Sentences: ', tokenized)

    wordCount = []                                                  # Keeps track of # of words in each sentence; Use for para interval extension
    for sentence in tokenized:                                      # For Each sentence in the text:
        totalWordCount = len(sentence.split())                      # Split the sentence on spaces and count # of words
        if len(wordCount) > 0:                                      # If sentence idx > 0,
            wordCount.append(totalWordCount+wordCount[-1])          # Add current word count with word count of setence previous
        else:           
            wordCount.append(totalWordCount)                        # Else, append normally
    return wordCount

# ----------------------------------- # Tuples (Eq#, Idx#) -----------------------------------
# Description: Creating an array of tuples (equation #, line number) 
# @Param output = Array of strings/words of original HTML doc
# --------------------------------------------------------------------------------------------
def eqTuples(output):
    eqno = []
    count = 1
    # Checking for equations + line number
    for i in range(len(output)):
        if output[i] == 'mathmarker':          # There is a block equation
            eqno.append([count, i+1])                            # i+2 since i = mathequation, i+1 = equation location, i+2 = equation #
            count += 1
    return eqno

# ------------------------------ # Starting Paragraph Intervals ------------------------------
# Description: Outputs initial paragraph interval index into (Eq #, idx #) output array 
# @Param    eqno = Tuples of (Eq#, Idx# of Eq#)
#           output = Array of strings/words of original HTML doc
# --------------------------------------------------------------------------------------------
def startInterval(eqno, output):
    paraBreak = []                                      # New array with paragraph breaks
    counter = 0                                         # Counter for current Word in PDF
    temp = 0                                            # Placeholder for latest occurence of a paragraph break before equation
    paragraph = 'parabreak'                             # Marker placed to locate paragraph breaks
    for i in range(len(eqno)):                          # Iterating through (Eq, idx number) pairs
        for idx in range(counter, eqno[i][1]-1):        # Iterating through idx between previous Eq and current Eq
            currWord = output[idx]
            if paragraph == currWord:                   # If there is a parabreak marker...
                temp = idx                              # Set latest occurence of paragraph break
        paraBreak.append([(str(eqno[i][0])+'start'), temp])    # Append index to paragraph break list
        counter = eqno[i][1]                            # Set counter to start of next equation
        temp = eqno[i][1]                               # Set latest occurence of paragraph break to start of next equation
    return paraBreak

# ------------------------------------ # End of Interval -------------------------------------
# Description: Extend range of text from end of equation to one sentence after
# @Param    eqno = Tuples of (Eq#, Idx# of Eq#)
#           wordCount = # of words per sentence
# --------------------------------------------------------------------------------------------
def endInterval(eqno, wordCount):
    exten = []                                                      # List for holding the chunks of text after the equation
    for idx, eqNum in enumerate(eqno):                              # Index and (eq#, idx#) pair
        startIdx = eqNum[1]                                         # Start of the portion of text AFTER the equation
        wordIDX = 0                                                 # Counter for idx of wordCount array
        while wordCount[wordIDX] < startIdx:                        # Iterate through wordCount array until total words exceed current index (startIdx)
            wordIDX +=1                                             # Interval will go one more then necessary so
        sentenceEndIdx = wordCount[wordIDX]                         # Set end interval to wordCount[wordIDX-1]
        exten.append([str(eqno[idx][0])+'end', sentenceEndIdx+10])   # Append current index as end of section
    return exten
  

# ----------------------------------------- # Main -------------------------------------------
# Description: Call all functions here
# --------------------------------------------------------------------------------------------

def main():
    url = 'file:///C:/Users//Desktop/MLP/Derivation-Tree/articles/1409.0466.html'      # Original Mathematical Document
    mathML, text = eqExtract(url)                   # Extract Block Equations, text holds processed HTML
    # print(text)
    eqIDs = idExtract(mathML)                       # Extract all Block Equation IDs
    print(eqIDs)
    wordCount = sentenceCount(text)                 # Calculate # of words per sentence
    # print('# of Words per sentence: ', wordCount)
    stringArr = arrOfStrings(text)                  # Convert text to array of strings
    # print('Text to Array of Strings: ', stringArr)
    equations = eqTuples(stringArr)                 # Create Tuples of (Eq#, Idx#)
    start = startInterval(equations, stringArr)     # Start Paragraph interval per equation
    end = endInterval(equations, wordCount)         # End Paragraph interval per equation
    print("Paragraph breaks: ", start)
    print("No Paragraph breaks: ", equations)
    print("Paragraph extension: ", end)
    # adjList = subTreeSimilarity(equations, start, stringArr, mathML, end)         # Returns adjacency list of outputted PNG
    adjList = bruteForce_Segmentation(equations, start, stringArr, mathML, end)     # Returns adjacency list of outputted PNG

if __name__ == "__main__":
    main()

# ----------------------------------- Part-Of-Speech-Tagging -----------------------------------

'''
for i in tokenized:
     
    # Word tokenizers is used to find the words
    # and punctuation in a string
    wordsList = nltk.word_tokenize(i)
 
    # Using a Tagger. Which is part-of-speech
    # tagger or POS-tagger.
    tagged = nltk.pos_tag(wordsList)
 
    print(tagged)
'''

# ----------------------------------- Debugging Subtree Similarity -----------------------------------

'''
# Parse all block/numbered equations within Math paper
results = soup.findAll("math", {"display" : "block"})

# Test Trees
string1 = str(results[0])
string2 = str(results[1])
string3 = str(results[2])
string4 = str(results[3])
string5 = str(results[4])
string6 = str(results[5])
string7 = str(results[6])


# Convert into OP Trees
root1 = toOpTree(string1)
root2 = toOpTree(string2)
root3 = toOpTree(string3)
root4 = toOpTree(string4)
root5 = toOpTree(string5)
root6 = toOpTree(string6)
root7 = toOpTree(string7)

# Print in order traversal of Tree; For Debugging
# IOT(root2)
'''

# ----------------------------------- Array of Tuples (Eq#, Idx#) -----------------------------------

'''
# Creating an array of (equation #, line number) pairs
eqno = []
idx = 1                                 # All equations start from 1
asc = 97                                # Ascii for 'a'
# Checking for equations + line number
for i in range(len(output)):
    temp = str(idx)                     # Equation Number
    tempascii = str(idx) + chr(asc)     # Equation Number w/ subequation
    nextTemp = str(idx+1)               # Next Equation Number 
    nextAscii = str(idx+1) + 'a'        # Next Equation Number w/ subequation
    if i >= 1 and temp in output[i] and output[i-1] == 'mathequation':          # Equation is regular
        eqno.append([idx, i])
        idx += 1
        continue
    if i >= 1 and tempascii in output[i] and output[i-1] == 'mathequation':     # Equation has subequation
        eqno.append([str(idx)+chr(asc), i])
        asc += 1
        continue
    if i >= 1 and nextTemp in output[i] and output[i-1] == 'mathequation':      # Next equation no longer has a, b, c etc.
        eqno.append([idx+1, i])
        idx += 2
        asc = 97
        continue
    if i >= 1 and nextAscii in output[i] and output[i-1] == 'mathequation':     # Next equation moves onto next idx w/ subequation
        eqno.append([str((idx+1))+'a', i])
        idx += 2
        asc = 98
        continue
# Debugging for eqno
# print('Equation # + Index Pair: ', eqno)
'''

# ------------------------------- # False Positive/Negatives --------------------------------
# Description: Identifies all false positives/negatives in a Mathematical Document
# @Param    url = Link to the Mathematical Document
#           adjList = Adjacency List that was created from the tempGraphing file
#           derivEdge = Dictionary that holds all correct edges
# --------------------------------------------------------------------------------------------

'''
def PosNeg(url, adjList, derivEdge):

    edge = []
    falseNeg = []
    falsePos = []
    correctUrl = derivEdge[url]

    for key in adjList.getKeys():                 # Converting Dictionary to array of edge tuples
        for val in adjList.getEdges(key):
            edge.append((int(key), int(val)))
    
    for tuple in correctUrl:                # Checking for False Negatives
        if tuple not in edge:
            falseNeg.append(tuple)
        if tuple in edge:               
            edge.remove(tuple)
    
    falsePos = edge                         # Remaining edges are false positives

    if len(falseNeg) >= 1:
        print('False Negative Edges: ', falseNeg)

    if len(falsePos) >= 1:
        print('False Positive Edges: ', falsePos)

'''
