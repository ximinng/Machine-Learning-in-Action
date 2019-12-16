# -*- coding: utf-8 -*-
"""
   Description : 
   Author :        xxm
"""
import pickle

"""写入"""
f = open('data_one.pkl', 'wb')

data_dict = {'name': 'Bruce', 'age': 25, 'high': 175}
train_list = [
    '(CNN) A dramatic picture of a brooding storm engulfing boats in a sailing race on Lake Geneva has earned Swiss photographer Loris von Siebenthal the Mirabaud Yacht Racing Image award for 2019.\n\nThe stunning photo, voted for by an international jury, beat entries from 132 other photographers to win the 10th edition of the competition.\n\nVon Siebenthal, who captured his picture during the 81st Bol d\'Or Mirabaud in June, said it was a "great honor" to receive what he called sailing photography\'s "absolute reference."\n\nAmerican Sharon Green came second with her shot of Volvo 70 Wizard off Miami in the Pineapple Cup, while British photographer Ian Roman\'s picture of Team USA\'s giant catamaran capsizing off Cowes, UK in the SailGP series came in third.\n\n"When we are on the water, our job is to anticipate visually challenging situations," Von Siebenthal said as he picked up the award at the Yacht Racing Forum in Bilbao, Spain.\n\nRead More',
    '(CNN) A dramatic picture of a brooding storm engulfing boats in a sailing race on Lake Geneva has earned Swiss photographer Loris von Siebenthal the Mirabaud Yacht Racing Image award for 2019.\n\nThe stunning photo, voted for by an international jury, beat entries from 132 other photographers to win the 10th edition of the competition.\n\nVon Siebenthal, who captured his picture during the 81st Bol d\'Or Mirabaud in June, said it was a "great honor" to receive what he called sailing photography\'s "absolute reference."\n\nAmerican Sharon Green came second with her shot of Volvo 70 Wizard off Miami in the Pineapple Cup, while British photographer Ian Roman\'s picture of Team USA\'s giant catamaran capsizing off Cowes, UK in the SailGP series came in third.\n\n"When we are on the water, our job is to anticipate visually challenging situations," Von Siebenthal said as he picked up the award at the Yacht Racing Forum in Bilbao, Spain.\n\nRead More',
]
test_list = ['test one', 'test two']

pickle.dump(
    {
        'dict_': data_dict,
        'train_': train_list,
        'test_': test_list
    }, f)
f.close()

"""读取"""
fr = open('data_one.pkl', 'rb')
c = pickle.load(fr)
fr.close()

print(c['dict_'])
print(c['train_'])
print(c['test_'])
