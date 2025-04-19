# testcases.py

TEST_CASES = [
    {
        # Basic test: single tweet and getNewsFeed.
        "operations": ["Twitter", "postTweet", "getNewsFeed"],
        "arguments": [[], [1, 5], [1]],
        "expected": [None, None, [5]]
    },
    {
        # Test following and unfollowing:
        # - User 1 posts a tweet.
        # - User 2 posts a tweet.
        # - User 1 follows User 2.
        # - User 2 posts another tweet.
        # - getNewsFeed(1) should return tweets from both users (most recent first).
        # - After unfollow, getNewsFeed(1) should return only User 1's tweet.
        "operations": ["Twitter", "postTweet", "postTweet", "follow", "postTweet", "getNewsFeed", "unfollow", "getNewsFeed"],
        "arguments": [[], [1, 1], [2, 2], [1, 2], [2, 3], [1], [1, 2], [1]],
        "expected": [None, None, None, None, None, [3, 2, 1], None, [1]]
    },
    {
        # Test multiple follow/unfollow operations:
        # - User 1 posts a tweet.
        # - User 1 follows User 2.
        # - User 2 posts a tweet.
        # - User 1 unfollows User 2.
        # - User 2 posts another tweet.
        # - getNewsFeed(1) should return only User 1's tweet.
        "operations": ["Twitter", "postTweet", "follow", "postTweet", "unfollow", "postTweet", "getNewsFeed"],
        "arguments": [[], [1, 10], [1, 2], [2, 20], [1, 2], [2, 30], [1]],
        "expected": [None, None, None, None, None, None, [10]]
    },
    {
        # Test with multiple users and follow operations:
        # - User 1 posts a tweet.
        # - User 2 posts a tweet.
        # - User 3 follows both User 1 and User 2.
        # - User 1 posts another tweet.
        # - getNewsFeed(3) should return tweets from User 1 and User 2 in descending order.
        "operations": ["Twitter", "postTweet", "postTweet", "follow", "follow", "postTweet", "getNewsFeed"],
        "arguments": [[], [1, 5], [2, 6], [3, 1], [3, 2], [1, 7], [3]],
        "expected": [None, None, None, None, None, None, [7, 6, 5]]
    },
    {
        # Example test case provided in the problem statement.
        "operations": [
            "Twitter", "postTweet", "getNewsFeed",
            "follow", "postTweet", "getNewsFeed",
            "unfollow", "getNewsFeed"
        ],
        "arguments": [
            [], [1, 5], [1],
            [1, 2], [2, 6], [1],
            [1, 2], [1]
        ],
        "expected": [
            None, None, [5],
            None, None, [6, 5],
            None, [5]
        ]
    },
]
