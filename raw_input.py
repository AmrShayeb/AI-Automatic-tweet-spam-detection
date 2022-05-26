
class Data:

	def __init__(self, twitterID, tweetContent, following, followers, actions, isretweet, location, Type, tweetContentLength, hasURL, similarity, finaltweetContent ):
			self.__twitterID = twitterID
			self.__tweetContent = tweetContent
			self.__following = following
			self.__followers = followers
			self.__actions = actions
			self.__isretweet = isretweet
			self.__location = location
			self.__Type = Type
			self.__tweetContentLength = tweetContentLength
			self.__hasURL = hasURL
			self.__similarity = similarity
			self.__finaltweetContent = finaltweetContent
			self.__dataVector = [[]]
			self.__perCapWords = 0
					


    # getters

	def gettwitterID(self):
        	return self.__twitterID

	def gettweetContent(self):
        	return self.__tweetContent

	def getfollowers(self):
        	return self.__followers

	def getfollowing(self):
        	return self.__following

	def getactions(self):
        	return self.__actions

	def getisretweet(self):
        	return self.__isretweet

	def getlocation(self):
					return self.__location

	def gettype(self):
        	return self.__Type

	def gettweetContentLength(self):
        	return self.__tweetContentLength

	def getsimilarity(self):
        	return self.__similarity

	def getfinaltweetContent(self):
        	return self.__finaltweetContent

	def getHasURL(self):
        	return self.__hasURL


	def getDataVector(self):
        	return self.__dataVector

	def getPerCapWords(self):
        	return self.__perCapWords


    # setter

	def settwitterID(self, twitterID):
        	self.__twitterID = twitterID

	def settweetContent(self, tweetContent):
        	self.__tweetContent = tweetContent

	def setfollowers(self, followers):
        	self.__followers =followers

	def setfollowing(self, following):
        	self.__following = following

	def setactions(self, actions):
        	self.__actions = actions

	def setisretweet(self, isretweet):
        	self.__isretweet = isretweet

	def setlocation(self, location):
					self.__location = location
	def settype(self, Type):
					self.__Type = Type

	def settweetContentLength(self, tweetContentLength):
        	self.__tweetContentLength = tweetContentLength

	def setSimilarity(self, similarity):
        	self.__similarity = similarity

	def setfinaltweetContent(self, finaltweetContent):
        	self.__finaltweetContent = finaltweetContent

	def setHasURL(self, hasURL):
        	self.__hasURL = hasURL


	def setDataVector(self, dataVector):
        	self.__dataVector = dataVector


	def setPerCapWords(self, perCapWords):
        	self.__perCapWords = perCapWords