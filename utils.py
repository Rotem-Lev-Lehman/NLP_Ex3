import re


def contains_URL(text):
    """ Checks whether the given text contains a URL.

    :param text: the text we want to check
    :type text: str
    :return: 1 if the text contains a URL, else 0
    :rtype: int
    """
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex,text)
    return int(len([x[0] for x in url]) > 0)


def count_caps_locked_words(text):
    """ Counts the amount of caps locked words in the given text.

    :param text: the text we want to check
    :type text: str
    :return: the amount of caps locked words in the given text
    :rtype: int
    """
    caps_locked_num = sum(map(str.isupper, text.split()))
    return caps_locked_num


def count_hashtags(text):
    """ Counts the amount of hashtags(#) in the given text.

    :param text: the text we want to check
    :type text: str
    :return: the amount of hashtags(#) in the given text
    :rtype: int
    """
    hashtags_num = text.count('#')
    return hashtags_num


def count_mentions(text):
    """ Counts the amount of mentions of other tweeter users(@) in the given text.

    :param text: the text we want to check
    :type text: str
    :return: the amount of mentions of other tweeter users(@) in the given text
    :rtype: int
    """
    mentions_num = text.count('@')
    return mentions_num
