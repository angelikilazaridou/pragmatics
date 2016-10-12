'''
Created on Mar 7, 2016

@author: thenghiapham
'''

import sys

head = """<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
<!--link type="text/css" rel="stylesheet" href="demo/candc.css"-->
</head>
<body>
<tr>
Model file: model_f <br/>
</tr>
"""

tail = """
</body>
</html>
"""


line_template = """
<div class="pair">
Symbol: symbol_v
<table width=60%>
  <tr>
    <td width=30%>
     target<br/>
      <img src="http://clic.cimec.unitn.it/~angeliki.lazaridou/agents/objects/id_1.jpg" style="height:228px;">
    </td>
    <td width=30%>
      context<br/>
      <img src="http://clic.cimec.unitn.it/~angeliki.lazaridou/agents/objects/id_2.jpg" style="height:228px;">
    </td>
  </tr>
</table>
</div>
<br/>
<br/>
"""
def txt2html(input_file, output_file):
    with open(input_file, "r") as i_stream, open(output_file, "w") as o_stream:
        i = 0
        for line in i_stream:
            i = i + 1
            line = line.strip()
	    if i==1:
	        tags = head.replace("model_f",line)
	    else:
                tags = line2tag(line)
            o_stream.write(tags)
        o_stream.write(tail)    

def line2tag(line):
    els = line.split(" ")
    symbol = els[0]
    target = els[1]
    id_1 = els[2]
    context= els[3]
    id_2 = els[4]
    ret_s = line_template.replace("symbol_v", symbol)
    ret_s = ret_s.replace("target", target).replace("context", context)
    ret_s = ret_s.replace("id_1", id_1).replace("id_2", id_2)
    return ret_s
    
if __name__ == "__main__":
    input_file = sys.argv[1]
    if len(sys.argv)>2:
        output_file = sys.argv[2]
    else:
	output_file = input_file+".html"
    txt2html(input_file, output_file)

