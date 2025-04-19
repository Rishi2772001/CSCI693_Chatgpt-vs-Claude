import collections
import heapq
from typing import List

class Twitter:
    def __init__(self):
        """
        Initialize the Twitter object.
        """
        self.followMap = collections.defaultdict(set)  # userId -> set of followeeIds
        self.tweetMap = collections.defaultdict(list)  # userId -> list of [timestamp, tweetId]
        self.timestamp = 0  # Global timestamp for ordering tweets
        
    def postTweet(self, userId: int, tweetId: int) -> None:
        """
        Compose a new tweet with ID tweetId by the user userId.
        Each call to this function will be made with a unique tweetId.
        """
        self.tweetMap[userId].append([self.timestamp, tweetId])
        self.timestamp -= 1  # Decrementing to ensure most recent tweets have higher priority
        
    def getNewsFeed(self, userId: int) -> List[int]:
        """
        Retrieve the 10 most recent tweet IDs in the user's news feed.
        Each item in the news feed must be posted by users who the user followed or by the user themselves.
        Tweets must be ordered from most recent to least recent.
        """
        # Initialize the result
        result = []
        
        # Create a min heap to merge tweets
        minHeap = []
        
        # Add the user to their own feed
        self.followMap[userId].add(userId)
        
        # Collect tweets from all users that userId follows
        for followeeId in self.followMap[userId]:
            if followeeId in self.tweetMap:
                # Get the index of the most recent tweet
                idx = len(self.tweetMap[followeeId]) - 1
                # If this user has tweets, add the most recent one to the heap
                if idx >= 0:
                    timestamp, tweetId = self.tweetMap[followeeId][idx]
                    # We use a min heap but want the most recent tweets, so we negate the timestamp
                    heapq.heappush(minHeap, [timestamp, tweetId, followeeId, idx - 1])
        
        # Get the 10 most recent tweets
        while minHeap and len(result) < 10:
            timestamp, tweetId, followeeId, idx = heapq.heappop(minHeap)
            result.append(tweetId)
            
            # If there are more tweets from this user, add the next one to the heap
            if idx >= 0:
                nextTimestamp, nextTweetId = self.tweetMap[followeeId][idx]
                heapq.heappush(minHeap, [nextTimestamp, nextTweetId, followeeId, idx - 1])
        
        return result
        
    def follow(self, followerId: int, followeeId: int) -> None:
        """
        The user with ID followerId started following the user with ID followeeId.
        """
        self.followMap[followerId].add(followeeId)
        
    def unfollow(self, followerId: int, followeeId: int) -> None:
        """
        The user with ID followerId started unfollowing the user with ID followeeId.
        """
        # Make sure we don't remove the user from their own feed
        if followerId != followeeId and followeeId in self.followMap[followerId]:
            self.followMap[followerId].remove(followeeId)