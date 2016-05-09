Website Fingerprinting - Analysis of Traffic Analysis attack and counter measures (Course 6350 : Big Data Analytics, Prof. Dr. Latifur Khan, University of Texas at Dallas)
=================================================================================

A web user today has a trillion web pages seeking his attention every time he is online with thousands of websites being added each day. 
Owing to the decentralized and unregulated nature of issuance of domain names and the low cost of web hosting, it has become extremely 
simple for anyone to buy a domain name and host a website on his own. 
This strength for the growth of internet, has also meant that anyone can use the web to create a fraudulent website with content and URL’s 
stolen from existing genuine websites.


Implementation 
==========================
Inspired by the extensive research work on website fingerprinting attacks of Dr. K.P. Dyer 
website: https://kpdyer.com
https://github.com/kpdyer/website-fingerprinting

We developed an improvement over the existing code in the form of a Sliding Window approach to change the temporally varying test sets 
and produce a model that trains with time and provides a model to give consistent accuracy. Major achievement were the implementation 
of Sliding test-set window and creation of a Ensemble o Classifiers.

Project Members 
==========================
Bikramjit Mandal,
Swapnil Shah,
Divya Varma,
Aruksha Grover,
Saloni Karnik,
Saurabh Chaudhary,
Shashank Buch,
Tushar Bhatia


Traffic Analysis Framework
==========================

This is a Python framework to compliment "Peek-a-Boo, I Still See You: Why Efficient Traffic Analysis Countermeasures Fail" [1].


References
----------
* [1] Dyer K.P., Coull S.E., Ristenpart T., Shrimpton T. Peek-a-Boo, I Still See You: Why Efficient Traffic Analysis Countermeasures Fail, To appear at IEEE Security and Privacy 2012
* [2] Marc Liberatore and Brian Neil Levine, Inferring the Source of Encrypted HTTP Connections. Proceedings of the 13th ACM Conference on Computer and Communications Security (CCS 2006)
* [3] Dominik Herrmann, Rolf Wendolsky, and Hannes Federrath. Website Fingerprinting: Attacking Popular Privacy Enhancing Technologies with the Multinomial Naive-Bayes Classiﬁer. In Proceedings of the ACM Workshop on Cloud Computing Security, pages 31–42, November 2009.