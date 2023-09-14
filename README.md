# Real-Time-Clustering
World News Stream : Kafka, flink, python, Java, Spring, ELK, Bigdata Handling (option - Grafana)





MORE INFO. : \
https://juny2312.github.io/generic2-1.html \
Unrevealed Code yet \
Would be revealed with DEMO !




       +------------------------+
       |       test_user        |
       +------------------------+
       | userID (PK)   char(17) |
       | name          varchar(15)|
       | addr          char(20)  |
       | mobile        char(3)   |
       | mdate         date      |
       +------------------------+
             |
             |
             | 1
             |
             |
       +------------------------+
       |    nasdaqPrice         |
       +------------------------+
       | num (PK)      int      |
       | tickerID (FK) char(20) |
       | price         int      |
       +------------------------+
             |
             |
             | M
             |
             |
       +------------------------+
       |     kospiPrice         |
       +------------------------+
       | num (PK)      int      |
       | tickerID (FK) char(20) |
       | price         int      |
       +------------------------+
             |
             |
             | M
             |
             |
       +------------------------+
       |    kosdaqPrice         |
       +------------------------+
       | num (PK)      int      |
       | tickerID (FK) char(20) |
       | price         int      |
       +------------------------+
             |
             |
             | 1
             |
             |
       +------------------------+
       |     newsDesk           |
       +------------------------+
       | num (PK)      int      |
       | newsID (FK)   char(20) |
       | newsTitle     char(20) |
       | newsDes       varchar(1500) |
       +------------------------+


