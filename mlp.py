
import csv
import matplotlib.pyplot as plt
from random import shuffle
from numpy import zeros
from numpy import tanh
from numpy import exp
from numpy import arctan
from numpy import log

#                        define activation functions    

def sig(x):
    return 1/(1 + exp(-x))

def dsig(x):
    return sig(x)*(1-sig(x))

def dtanh(x):
    return 1-tanh(x)*tanh(x)

def relu(x):
	return max(0.0, x)

def drelu(x):
    return 1

def identity(x):
    return x

def didentity(x):
    return 1

def binarystep(x):
    if x<0:
        return 0
    else:
        return 1

def dbinarystep(x):
    return 0

def darctan(x):
    return 1/((x**2)+1)

def softplus(x):
    return log(1+exp(x))

def dsoftplus(x):
    return 1/(1 + exp(-x))

#                        taking dataset and define number of input and output layer neurons  
       
training=(float(input('\n\nHow much percentage of your dataset is allocated for training ?  ')))/100
biggest_error=100-(float(input('\n\nPlease enter the accuracy of the network :  ')))
weight_initialize = float(input('\n\nPlease enter a value for weight initialize :  '))
bias_initialize = float(input('\n\nPlease enter a value for bias initialize :  '))
alpha=float(input('\n\nPlease enter a value for learning rate :  '))
layers=[]
xcolumn=[]
tcolumn=[]
inputs=[]
targets=[]
indexcounter=0
linecounter=0

with open('C:/Users/HOSEIN/Desktop/NNP/dataset.csv') as csvfile:
    data = csv.reader(csvfile)

    for line in data:
        
        if linecounter==0:
            
            for member in line:
                
                if member[0]=='x':
                   xcolumn.append(indexcounter)
                   indexcounter=indexcounter+1
                    
                elif member[0]=='t':
                    tcolumn.append(indexcounter)
                    indexcounter=indexcounter+1 
            
            linecounter = linecounter+1
            layers.append(len(xcolumn))
            layers.append(len(tcolumn))
            
        elif linecounter>2:
            indexcounter=0
            line_inputs=[]
            line_targets=[]
            
            for member in line:
                
                if indexcounter in xcolumn:
                    line_inputs.append(float(member))
                    
                elif indexcounter in tcolumn:
                    line_targets.append(float(member))
                
                indexcounter = indexcounter+1
                
            inputs.append(line_inputs)
            targets.append(line_targets)
        
        linecounter = linecounter+1      

#print(len(inputs))
#print(targets)

#                        normalize inputs and targets

nor=[]

for i in range(len(inputs)):
    
    for j in range(len(inputs[i])):
        
        nor.append(inputs[i][j])

for i in range(len(targets)):
    
    for j in range(len(targets[i])):
        
        nor.append(targets[i][j])


#                        normalize inputs
        
for i in range(len(inputs)):
    
    for j in range(len(inputs[i])):
        
        inputs[i][j] = (inputs[i][j] - min(nor)) / (max(nor) - min(nor))

# print(inputs)

#                        normalize targets 
     
for i in range(len(targets)):
    
    for j in range(len(targets[i])):
        
        targets[i][j] = (targets[i][j] - min(nor)) / (max(nor) - min(nor))

target_list=[]

for i in range(len(targets)):
    
    for j in range(len(targets[i])):
        
        target_list.append(targets[i][j])

sorted_targetlist = sorted(target_list)
second_lowest = sorted_targetlist[1]

for i in range(len(targets)):
    
    for j in range(len(targets[i])):
        
        if targets[i][j]==0:
            targets[i][j] = second_lowest/(10**6)

print(targets)
       
#                        create a list of random indexes

random_index = [i for i in range(len(inputs))]
shuffle(random_index)

#                        taking activation functions and define number of hidden layers

f={}
linecounter=1

with open('C:/Users/HOSEIN/Desktop/NNP/hiddenlayer.csv') as csvfile:
    data = csv.reader(csvfile)

    for line in data:
        
        if line[0]=='outputlayer':
            f[linecounter] = line[1]
            
        else:
            layers.insert(linecounter,int(line[1]))
            f[linecounter] = line[0]
            linecounter = linecounter+1

# print(layers)

#                        define the derivative of the avtivation functions

df={}

for i in range(1,len(f)+1):  
                    
    if f[i]=='sig':
        df[i]='dsig'
        
    elif f[i]=='tanh':
        df[i]='dtanh'
        
    elif f[i]=='relu':
        df[i]='drelu'

    elif f[i]=='identity':
        df[i]='didentity'    

    elif f[i]=='binarystep':
        df[i]='dbinarystep'

    elif f[i]=='arctan':
        df[i]='darctan'

    elif f[i]=='softplus':
        df[i]='dsoftplus'

#                        create network matrix  

a=zeros((max(layers),len(layers)))

for i in range(len(layers)):
    for j in range(layers[i]):
        a[j][i]=1

# print(a)

#                       initializing and create dictionary for weights   

w={}

for i in range(len(layers)-1): # << i >> is counter for layer number

    for j in range(layers[i]): # << j >> is counter for neuron number
    
        w[str(i)+' '+str(j)]=zeros(layers[i+1]) + weight_initialize
            
# print('\n\n',w)

#                        initializing and create dictionary for biases       

b={}

for i in range(1,len(layers)):
    
    for j in range(layers[i]):
        
        b[str(i)+' '+str(j)] = bias_initialize

# print(b)

#
                                                        
training_epochs = 0
skip_counter = 0
z=0
x1=[]
y1=[]

while z < int(training*len(inputs)): #              training process

    #                        assign a random input and target to network for training

    for h in range(len(inputs[0])):
        a[h][0]=inputs[random_index[z]][h]
    
    t=targets[random_index[z]]
    
    #                        calculate forward path operations      
    
    net={}
    output={}
    
    for i in range(1,len(layers)):
    
        for j in range(layers[i]):
            
            net[str(i)+' '+str(j)]=0
            
            for k in range(layers[i-1]):
                
                net[str(i)+' '+str(j)] = net[str(i)+' '+str(j)] + ((w[str(i-1)+' '+str(k)][j])*(a[k][i-1]))
                
                if k==layers[i-1]-1:
                    
                    net[str(i)+' '+str(j)] = net[str(i)+' '+str(j)] + b[str(i)+' '+str(j)]
                    
            output[str(i)+' '+str(j)]=eval(f[i]+'('+str(net[str(i)+' '+str(j)])+')')
    
            a[j][i]=output[str(i)+' '+str(j)]
            
    # print(a)
    # print(b)
    # print(w)
    
    #                        calculate total error
    
    e={}
    
    for j in range(layers[-1]):
        
        e[str(j)]=(0.5)*(t[j]-a[j][len(layers)-1])**2
    
    Etotal=sum(e.values())
    
    # print(e)
    
    ##########################################################################
    #                            back propagation                            #
    ##########################################################################
    
    #                        calculate partial of Error to outputs
    
    error_output={}                                    
    Error=[]
    
    for j in range(len(e)):
        
        error_output[j]=a[j][len(layers)-1]-t[j]
        Error.append(abs((error_output[j]/t[j])*100))
       
    # print(error_output)
    print(Error)
    
    #

    if max(Error)>biggest_error:
    
        #                       calculate and routing from output  
        #                       neuron to selected weight for update
        
        skip_counter = 0
        
        for m1 in range(len(layers)-1):                      # m1 changes layer number of weights for calculate partial of output to weight
                                                             
            if m1<len(layers)-2:                             # for layers before layer one to last
        
                for m2 in range(layers[m1]):                 # m2 changes neuron number in layer with number m1
                
                    for m3 in range(layers[m1+1]):           # m3 changes weight number for neuron whid number m2 in layer with number m1 and 
                    
                        pew_list=[]
                        peb_list=[]
                        
                        for m4 in range(layers[-1]):         # m4 changes number of output neuron
                        
                            A=1
                            for i in range(len(layers)-3,m1+1,-1):                      # A is variable relative to layer change (m1) and number of network layers
                                A=A*layers[i]
                            
                            
                            if m1+3==len(layers):                                       # B is variable relative to layer change (m1) and number of network layers
                                B=1                          # for two layer before last layer
                            else:
                                B=layers[-2]
                                
                            C=len(layers)-(m1+2)                                        # C is variable relative to layer change (m1) and number of network layers
                            
                            M={}                             # M is a dictionary with rules of naming a matrix
                            
                            for i in range(layers[-2]):
                                
                                if m1+4==len(layers):                                   # for all lines of matrix elements in column number(i)(if there are two layers between the current layer(m1) and the output layer)
                                
                                    for j in range(A):                                  # first line of matrix element (line number zero)  
                                        M[str(j)+' '+str(i)+' '+str(0)]=str(len(layers)-2)+' '+str(i)+' '+str(m4)
                                        
                                    for j in range(A):                                  # for second line of matrix element(line number one)
                                        M[str(j)+' '+str(i)+' '+str(1)]=str(m1+1)+' '+str(m3)+' '+M[str(j)+' '+str(i)+' '+str(0)].split()[1]
                                
                                elif m1+3==len(layers):                                 # for the only line of matrix elements (if there is one layer between the current layer(m1) and the output layer)
                                    M[str(0)+' '+str(0)+' '+str(0)]=str(len(layers)-2)+' '+str(m3)+' '+str(m4)
                                
                                else:
                                    
                                    k=0
                                    for l2 in range(len(layers)-3,m1+1,-1):             # for all lines of matrix elements in column number(i) l2 is a counter for layer number in range(len(layers)-3,m1+1,-1)
                                        
                                        if k==0 :                                       # first line of matrix element (line number zero)
                                                                  
                                            for j in range(A):
                                                M[str(j)+' '+str(i)+' '+str(k)]=str(len(layers)-2)+' '+str(i)+' '+str(m4)
                                            
                                            k=k+1    
                                                   
                                        if l2!=m1+2 and k!=0 or m1+2==len(layers)-3:    # from second line of matrix element(line number one)
                                                                                        # to two line line before last line (line number C-3) and if
                                            l3=0                                        # l3 is a counter for neuron number in layer number l2
                                            for j in range(A):                          # (m1+2=len(layers)-3) to line before last line of matrix element
                                                
                                                if l3==layers[l2]:
                                                    l3=0
                                                M[str(j)+' '+str(i)+' '+str(k)]=str(l2)+' '+str(l3)+' '+M[str(j)+' '+str(i)+' '+str(k-1)].split()[1]
                                                
                                                l3=l3+1
                                            
                                            k=k+1
                                        
                                        if l2==m1+2 and k!=0 or m1+2!=len(layers)-3:    # line before last line of matrix element and if
                                            l3=0                                        # (m1+2!=len(layers)-3) this condition will be enforced
                                            counter=0
                                            
                                            for j in range(A):
                                                
                                                if counter==layers[l2]:
                                                    counter=0
                                                    l3=l3+1
                                                    
                                                    if l3==layers[l2]:
                                                        l3=0
                                                
                                                M[str(j)+' '+str(i)+' '+str(k)]=str(l2)+' '+str(l3)+' '+M[str(j)+' '+str(i)+' '+str(k-1)].split()[1]
                                                counter=counter+1
                                        
                                        if l2==m1+2:                                    # last line of matrix element (line number C-1)                                                              
                                            k=C-1
                                            
                                            for j in range(A):
                                                M[str(j)+' '+str(i)+' '+str(k)]=str(m1+1)+' '+str(m3)+' '+M[str(j)+' '+str(i)+' '+str(k-1)].split()[1]
                                                
                            
                            #                       print matrix of elements
                                   
                            # for i in range(B):
                                
                            #     for j in range(A):  
                                    
                            #         for k in range(C):
                            #             print(M[str(j)+' '+str(i)+' '+str(k)],'   ',k)
                                    
                            #         print('\n')
                            
                            
                            pow_list=[]
                            pob_list=[]
                                     
                            for i in range(B):
                                
                                for j in range(A):              # calculate derivative of otputlayer * a[m2][m1]            
                                    p_outm4_weightm1m2m3 = a[m2][m1] * eval(df[len(layers)-1]+ '(' + str(net[str(len(layers)-1) + ' ' + str(m4)])+ ')')
                                    
                                    if m2==0:                   # calculate derivative of otputlayer         
                                        p_outm4_biasm1m2 = eval(df[len(layers)-1]+ '(' + str(net[str(len(layers)-1) + ' ' + str(m4)])+ ')')
                                    
                                    for k in range(C):          # calculate partial of output to weight in above line
                                        p_outm4_weightm1m2m3 = p_outm4_weightm1m2m3 * (w[M[str(j)+' '+str(i)+' '+str(k)].split()[0] + ' ' + M[str(j)+' '+str(i)+' '+str(k)].split()[1]][int(M[str(j)+' '+str(i)+' '+str(k)].split()[2])]) * (eval(df[float(M[str(j)+' '+str(i)+' '+str(k)].split()[0])] + '(' + str(net[M[str(j)+' '+str(i)+' '+str(k)].split()[0] + ' ' + M[str(j)+' '+str(i)+' '+str(k)].split()[1]]) + ')'))    
                                        
                                        if m2==0:               # calculate partial of output to bias in above line
                                            p_outm4_biasm1m2 = p_outm4_biasm1m2 * (w[M[str(j)+' '+str(i)+' '+str(k)].split()[0] + ' ' + M[str(j)+' '+str(i)+' '+str(k)].split()[1]][int(M[str(j)+' '+str(i)+' '+str(k)].split()[2])]) * (eval(df[float(M[str(j)+' '+str(i)+' '+str(k)].split()[0])] + '(' + str(net[M[str(j)+' '+str(i)+' '+str(k)].split()[0] + ' ' + M[str(j)+' '+str(i)+' '+str(k)].split()[1]]) + ')'))
                                    
                                    pow_list.append(p_outm4_weightm1m2m3)
                                    
                                    if m2==0:
                                        pob_list.append(p_outm4_biasm1m2)
                            
                            p_outm4_weightm1m2m3=sum(pow_list)
                            p_errorm4_weightm1m2m3 = error_output[m4] * p_outm4_weightm1m2m3
                            pew_list.append(p_errorm4_weightm1m2m3)

                            if m2==0:
                                p_outm4_biasm1m2=sum(pob_list)
                                p_errorm4_biasm1m2 = error_output[m4] * p_outm4_biasm1m2
                                peb_list.append(p_errorm4_biasm1m2)
                                
                        p_Etotal_weigthm1m2m3 = sum(pew_list)                
                        w[str(m1)+' '+str(m2)][m3] = w[str(m1)+' '+str(m2)][m3] - (alpha * p_Etotal_weigthm1m2m3)
                        
                        if m2==0:
                            p_Etotal_biasm1m2 = sum(peb_list)
                            b[str(m1+1)+' '+str(m3)] = b[str(m1+1)+' '+str(m2)] - (alpha * p_Etotal_biasm1m2)
                            
            elif m1 == len(layers)-2:

                for m2 in range(layers[m1]):
                    
                    for m3 in range(layers[m1+1]):
                        p_Etotal_weightm1m2m3 = error_output[m3] * eval(df[len(layers)-1] + '(' + str(net[str(len(layers)-1) + ' ' + str(m3)])+ ')') * a[m2][m1]
                        w[str(m1)+ ' ' +str(m2)][m3] = w[str(m1)+ ' ' +str(m2)][m3] - (alpha * p_Etotal_weightm1m2m3)
                        
                        if m2==0:
                            p_Etotal_biasm1m2 = error_output[m3] * eval(df[len(layers)-1] + '(' +str(net[str(len(layers)-1) +' '+str(m3)])+ ')')
                            b[str(m1+1)+ ' ' +str(m2)] = b[str(m1+1)+ ' ' +str(m2)] - (alpha * p_Etotal_biasm1m2)
                                                          
    else:                   
        skip_counter = skip_counter+1                       # skip_counter is a counter for samples that do not need to be updated
        # print('skipcounter = ',skip_counter)
    
    if skip_counter == int(training*len(inputs)):                           # for when we have left behind samples to the number one epoch and we have not updated yet
        z = int(training*len(inputs))
    
    if z==int((training*len(inputs))-1):
        training_epochs = training_epochs+1
        x1.append(training_epochs)
        y1.append(Etotal)
        print('training_epochs = ',training_epochs)
        z = -1
          
    z = z+1
                        
#

test_epochs = 0
x2=[]
y2=[]
outputs=[]

while z < len(inputs): #              test process  

    #                        assign a random input and target to network for training

    for h in range(len(inputs[0])):
        a[h][0]=inputs[random_index[z]][h]
    
    t=targets[random_index[z]]
    
    #                        calculate forward path operations      
    
    net={}
    output={}
    
    for i in range(1,len(layers)):
    
        for j in range(layers[i]):
            
            net[str(i)+' '+str(j)]=0
            
            for k in range(layers[i-1]):
                
                net[str(i)+' '+str(j)] = net[str(i)+' '+str(j)] + ((w[str(i-1)+' '+str(k)][j])*(a[k][i-1]))
                
                if k==layers[i-1]-1:
                    
                    net[str(i)+' '+str(j)] = net[str(i)+' '+str(j)] + b[str(i)+' '+str(j)]
                    
            output[str(i)+' '+str(j)]=eval(f[i]+'('+str(net[str(i)+' '+str(j)])+')')
    
            a[j][i]=output[str(i)+' '+str(j)]
            
            if i==len(layers)-1:
                outputs.append(a[j][i])
                    
    #                        calculate total error
    
    e={}
    
    for j in range(layers[-1]):
        
        e[str(j)]=(0.5)*(t[j]-a[j][len(layers)-1])**2
    
    Etotal=sum(e.values())
    
    test_epochs = test_epochs+1
    
    x2.append(test_epochs)
    
    y2.append(Etotal)
    
    # print('test_epochs = ',test_epochs)
    
    z = z + 1
 
print('\n\ninputs = ',b,'\n\noutputs = ',outputs,'\n\ntargets = ',targets)

#                       plot loss function

plt.style.use('default')
plt.plot(x1,y1,linewidth=0.75,label='Loss Function for Training Set',color='C3')
# plt.plot(x2,y2,linewidth=0.75,label='Loss Function for Validation Set',color='g')
plt.xlabel('EPOCH',fontsize=10)
plt.ylabel('J',fontsize=10)
plt.legend(fontsize=15,bbox_to_anchor=(1.05, -0.15),loc='best',)
plt.grid('on')
plt.savefig('C:/Users/HOSEIN/Desktop/NNP/LOSS FUNCTION.svg',format='svg',dpi=100000,bbox_inches='tight')
plt.show()



