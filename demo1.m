fin=fopen('./out/pred_s.txt','r');
result={};
while feof(fin)==0
    str=fgetl(fin);
    str=strread(str,'%s','delimiter',' \n');
    result=[result;str];
end
data_double=str2double(result);
c = size(data_double,1)/92;
data_double1 = reshape(data_double,92,c);
data_double1 =double(data_double1');
