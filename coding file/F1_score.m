function [score] = F1_score(X,theta,y)
z = X*(theta);
h = sigmoid(z);
h = round(h);
true_pos =0;
false_pos =0;
false_neg=0;

for i = 1:length(h)
  if h(i) ==1 && y(i)==1   
     true_pos = true_pos + 1;
  elseif( h(i) ==1 && y(i)==0)
      false_pos= false_pos +1;
      
  elseif( h(i) ==0 && y(i)==1   )
      false_neg= false_neg +1;
  end
  end

precision = true_pos/(true_pos+false_pos);
recall = true_pos/(true_pos + false_neg);

score = 2*precision*recall/(precision+recall);

end

