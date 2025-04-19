import heapq
from collections import defaultdict

class Twitter:
    def __init__(self):
        self.time = 0  # Global timestamp for ordering tweets.
        self.tweets = defaultdict(list)    # Maps userId -> list of (timestamp, tweetId)
        self.following = defaultdict(set)  # Maps userId -> set of followees

    def postTweet(self, userId: int, tweetId: int) -> None:
        # Ensure the user follows themselves.
        if userId not in self.following:
            self.following[userId].add(userId)
        else:
            self.following[userId].add(userId)
        # Append the tweet with the current timestamp.
        self.tweets[userId].append((self.time, tweetId))
        self.time += 1

    def getNewsFeed(self, userId: int) -> list:
        res = []
        heap = []
        # Ensure the user follows themselves.
        if userId not in self.following:
            self.following[userId].add(userId)
        # For every user followed (including self), add their most recent tweet.
        for followee in self.following[userId]:
            if self.tweets[followee]:
                idx = len(self.tweets[followee]) - 1  # Index of the most recent tweet.
                tweet_time, tweet_id = self.tweets[followee][idx]
                # Use negative timestamp to simulate a max-heap.
                heapq.heappush(heap, (-tweet_time, tweet_id, followee, idx))
        # Retrieve up to 10 most recent tweets.
        while heap and len(res) < 10:
            neg_time, tweet_id, followee, idx = heapq.heappop(heap)
            res.append(tweet_id)
            if idx - 1 >= 0:
                tweet_time, tweet_id2 = self.tweets[followee][idx - 1]
                heapq.heappush(heap, (-tweet_time, tweet_id2, followee, idx - 1))
        return res

    def follow(self, followerId: int, followeeId: int) -> None:
        # A user cannot follow themselves.
        if followerId == followeeId:
            return
        if followerId not in self.following:
            self.following[followerId].add(followerId)
        self.following[followerId].add(followeeId)

    def unfollow(self, followerId: int, followeeId: int) -> None:
        # A user cannot unfollow themselves.
        if followerId == followeeId:
            return
        if followerId in self.following:
            self.following[followerId].discard(followeeId)