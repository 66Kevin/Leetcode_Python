from collections import defaultdict

class Twitter:

    def __init__(self):
        self.tweets = defaultdict(list)
        self.following = defaultdict(list)
        self.timestamp = 0

    def postTweet(self, userId: int, tweetId: int) -> None:
        self.tweets[userId].append((tweetId, self.timestamp))
        if len(self.tweets[userId]) > 10:
            self.tweets[userId].pop(0)
        self.timestamp += 1

    def getNewsFeed(self, userId: int) -> List[int]:
        posts = self.tweets[userId][:]
        for following in self.following[userId]: # 遍历自己关注列表发过的twitter
            posts.extend(self.tweets[following]) # 把关注列表发过的twitter添加到list中
        posts.sort(key = lambda x : x[1])
        return [p[0] for p in posts[-10:]][::-1]

    def follow(self, followerId: int, followeeId: int) -> None:
        if followerId != followeeId and followeeId not in self.following[followerId]:
            self.following[followerId].append(followeeId)

    def unfollow(self, followerId: int, followeeId: int) -> None:
        if followeeId in self.following[followerId]:
            self.following[followerId].remove(followeeId)



# Your Twitter object will be instantiated and called as such:
# obj = Twitter()
# obj.postTweet(userId,tweetId)
# param_2 = obj.getNewsFeed(userId)
# obj.follow(followerId,followeeId)
# obj.unfollow(followerId,followeeId)
