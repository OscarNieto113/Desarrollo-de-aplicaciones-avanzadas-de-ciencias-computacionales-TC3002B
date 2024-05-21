from enum import IntEnum

class Tag(IntEnum):
	EOF = 65535
	ERROR = 65534
	## Operators ##
	GEQ = 258
	LEQ = 259
	NEQ = 260
	ASSIGN = 261
	## REGULAR EXPRESSIONS ##
	ID = 358
	NUMBER = 359
	STRING = 360
	TRUE = 361
	FALSE = 362
	## ADD THE MISSING RESERVED WORDS, JUST FOLLOW THE NUMBER SEQUENCE ##
	VAR = 457
	FORWARD = 458
	FD = 459
	BACKWARD = 460
	BK = 461
	RIGHT = 462
	RT = 463
	LEFT = 464
	LT = 465
	SETX = 466
	SETY = 467
	SETXY = 468
	HOME = 469
	CLEAR = 470
	CLS = 471
	CIRCLE = 472
	ARC = 473
	PENUP = 474
	PU = 475
	PENDOWN = 476
	PD = 477
	COLOR = 478
	PENWIDTH = 479
	PRINT = 480
	REPEAT = 481
	IF = 482
	IFELSE = 483
	OR = 484
	AND = 485
	MOD = 486
	
class Token:
	__tag = Tag.EOF
	__value = None
	
	def __init__(self, tagId, val = None):
		self.__tag = tagId
		self.__value = val
		
	def getTag(self):
		return self.__tag
	
	def getValue(self):
		return self.__value
		
	def __str__(self):
		if self.__tag == Tag.GEQ:
			return "'>='"
		elif self.__tag == Tag.LEQ:
			return "'<='"
		elif self.__tag == Tag.NEQ:
			return "'<>'"
		elif self.__tag == Tag.ASSIGN:
			return "':='"
		elif self.__tag == Tag.TRUE:
			return "'#t'"
		elif self.__tag == Tag.FALSE:
			return "'#f'"
		elif self.__tag == Tag.NUMBER:
			return "numeric constant"
		elif self.__tag == Tag.ID:
			return "'" + str(self.__value) + "'"
		elif self.__tag >= Tag.VAR and self.__tag <= Tag.MOD:
			return "'" +  str(self.__value).lower() + "'"
		elif self.__tag == Tag.STRING:
			return "string constant"
		else:
			return "'" + chr(self.__tag) + "'" 
			return "'" + chr(self.__tag) + "'" 
			
class Lexer:
	__peek = ' '
	__words = {}
	__input = None

	def __init__(self, filepath):
		self.__input = open(filepath, "r")
		self.__peek = ' '

		self.__words["VAR"] = Token(Tag.VAR, "VAR")
		self.__words["FORWARD"] = Token(Tag.FORWARD, "FORWARD")
		self.__words["FD"] = Token(Tag.FORWARD, "FD")
		self.__words["BACKWARD"] = Token(Tag.BACKWARD, "BACKWARD")
		self.__words["BK"] = Token(Tag.BACKWARD, "BK")
		self.__words["RIGHT"] = Token(Tag.RIGHT, "RIGHT")
		self.__words["RT"] = Token(Tag.RIGHT, "RT")
		self.__words["LEFT"] = Token(Tag.LEFT, "LEFT")
		self.__words["LT"] = Token(Tag.LEFT, "LT")
		self.__words["SETX"] = Token(Tag.SETX, "SETX")
		self.__words["SETY"] = Token(Tag.SETY, "SETY")
		self.__words["SETXY"] = Token(Tag.SETXY, "SETXY")
		self.__words["HOME"] = Token(Tag.HOME, "HOME")
		self.__words["CLEAR"] = Token(Tag.CLEAR, "CLEAR")
		self.__words["CLS"] = Token(Tag.CLEAR, "CLS")
		self.__words["CIRCLE"] = Token(Tag.CIRCLE, "CIRCLE")
		self.__words["ARC"] = Token(Tag.ARC, "ARC")
		self.__words["PENUP"] = Token(Tag.PENUP, "PENUP")
		self.__words["PU"] = Token(Tag.PENUP, "PU")
		self.__words["PENDOWN"] = Token(Tag.PENDOWN, "PENDOWN")
		self.__words["PD"] = Token(Tag.PENDOWN, "PD")
		self.__words["COLOR"] = Token(Tag.COLOR, "COLOR")
		self.__words["PENWIDTH"] = Token(Tag.PENWIDTH, "PENWIDTH")
		self.__words["PRINT"] = Token(Tag.PRINT, "PRINT")
		self.__words["REPEAT"] = Token(Tag.REPEAT, "REPEAT")
		self.__words["IF"] = Token(Tag.IF, "IF")
		self.__words["IFELSE"] = Token(Tag.IFELSE, "IFELSE")
		self.__words["OR"] = Token(Tag.OR, "OR")
		self.__words["AND"] = Token(Tag.AND, "AND")
		self.__words["MOD"] = Token(Tag.MOD, "MOD")

	def read(self):
		self.__peek = self.__input.read(1)
	
	def readch(self, c):
		self.read()
		if self.__peek != c:
			return False

		self.__peek = ' '
		return True

	def __skipSpaces(self):
		while True:
			if self.__peek == ' ' or self.__peek == '\t' or self.__peek == '\r' or self.__peek == '\n':
				self.read()
			else:
				break
	
	def scan(self):
		self.__skipSpaces()

		## ADD CODE TO SKIP COMMENTS HERE ##
		if self.__peek == '%':
			while self.__peek != '\n' and self.__peek != '':
				self.readch("")
			return self.scan()

		if self.__peek == '<':
			if self.readch('='):
				return Token(Tag.LEQ, "<=")
			elif self.readch('>'):
				return Token(Tag.NEQ, "<>")
			else:
				return Token(ord('<'))
		elif self.__peek == '>':
			if self.readch('='):
				return Word(Tag.GEQ, ">=")
			else:
				return Token(ord('>'))
		elif self.__peek == '#':
			if self.readch('t'):
				return Token(Tag.TRUE, "#t")
			elif self.readch('f'):
				return Token(Tag.FALSE, "#f")
			else:
				return Token(ord('#'))
		elif self.__peek == ':':
			if self.readch('='):
				#print("reading :=")
				return Token(Tag.ASSIGN, ":=")
			else:
				return Token(ord(':'))

		if self.__peek  == '"':
			val = ""
			while True:
				val = val + self.__peek
				self.read()
				if self.__peek == '"':
					break
			
			val = val + self.__peek
			self.read()
			return Token(Tag.STRING, val)

		if self.__peek.isdigit():
			val = 0
			while True:
				val = (val * 10) + int(self.__peek)
				self.read()
				if not(self.__peek.isdigit()):
					break
			## ADD CODE TO PROCESS DECIMAL PART HERE ##
			return Token(Tag.NUMBER, val)

		if self.__peek == '.':
			self.read()
			val = 0
			valnum = 10
			while True:
				val = (val + int(self.__peek)) / valnum
				valnum *= 10
				self.read()
				if not(self.__peek.isdigit()):
					break
			return Token(Tag.NUMBER, val)

		if self.__peek.isalpha():
			val = ""
			while True:
				val = val + self.__peek.upper()
				self.read()
				if not(self.__peek.isalnum()):
					break

			if val in self.__words:
				return self.__words[val]

			w = Token(Tag.ID, val)
			self.__words[val] = Token(Tag.ID, val)
			return w

		if not(self.__peek):
			return Token(Tag.EOF)			

		token = Token(ord(self.__peek))
		self.__peek = ' ' 
		return token
