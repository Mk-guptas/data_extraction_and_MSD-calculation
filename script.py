import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mtp
import pickle
import math
from scipy.optimize import curve_fit
from scipy.fft import fft , fftfreq ,ifft
from numpy import savetxt

mtp.rcParams['font.family']='Times New Roman'
frame=('1-505')
no_of_frame = 505
max_particle= 15000
conc=1 
foldername='1pernacl_8hr'
saving_location='D:/mk/python_analyzed_results/Nacl_experiments/5th march/'+foldername+'/dotpyfile'
frame_tracked=400

#reading locations and lable from a csv file
def reading_csv(source):
  a= pd.read_csv(source, usecols=['FilterObjects2_Location_Center_X'])
  x_location=(np.array(a))
  print(type(a))
  b=pd.read_csv(source, usecols=['FilterObjects2_Location_Center_Y'])
  y_location=np.array(b)
  img_no=pd.read_csv(source, usecols=['ImageNumber'])
  img_no =np.array(img_no)
  c=pd.read_csv(source, usecols=['FilterObjects2_TrackObjects_Label_50'])
  label=np.array(c)
  orientation=pd.read_csv(source, usecols=['FilterObjects2_AreaShape_Orientation'])
  orientation=np.asarray(orientation)                      
  return(x_location,y_location,label,orientation,img_no)
  #return(x_location,y_location,label)



#x_location,y_location, label,orientation= reading_csv('D:/mk/python_analyzed_results/Nacl_experiments/1stmarch/analyzed_images/2_nacl_8hr/analyzed_2per/tracking2pernaclPer_Object.csv')
#x_location,y_location,label = reading_csv('D:/mk/python_analyzed_results/Nacl_experiments/1stmarch/analyzed_images/0_nacl_8hr/analyzed_2min20secto33min/tracking02minacln20secPer_Object.csv')#   input file name


x_location,y_location, label,orientation,img_no= reading_csv('D:/mk/python_analyzed_results/Nacl_experiments/5th march/'+foldername+'/trackedimages1/trackeddata.csv')






#creating dictionary having list of list
def dictionary(no_of_list):
    name_of_dictionary={}
    for i in range(1,no_of_list):
     name_of_dictionary[i]=[]

    return(name_of_dictionary)
     
particle_x=dictionary(max_particle)
particle_y=dictionary(max_particle)
particle_orientation=dictionary(max_particle)
#print(particle_x)


#creating bins of a list
a=np.arange(0,30,1)
def create_bin(list,width_of_bin,upper_limit):
  binned=[]
  l=width_of_bin
  while l<= upper_limit:
    counter=0
    for i in (list):
        if l-width_of_bin <=i<l:
         counter=counter+1
    binned.append(counter)
    l=l+width_of_bin
  return(binned)

#binned=create_bin(a,1,30)
#print(len(binned))




#storing of locations of each particle in the dictionary
def update_dictionary():
 for i in range(len(label)):
    t=label[i]
    t3=(t[0])
    #print(t,t3)
       #change everytime                                        #only these particle are being tracked
    q2=x_location[i][0]
    q5=y_location[i][0]
    ori=orientation[i][0]
    time=img_no[i][0]
    #print(ori)
        # print(t3)
    particle_x[t3].append(q2)
    particle_y[t3].append(q5)
    particle_orientation[time].append(ori)
         
  #return (particle_x,particle_y)
 return (particle_x,particle_y,particle_orientation)

particle_x,particle_y,particle_orientation=update_dictionary()

filehandler500= open(saving_location+'/particle_orientation'+str(frame)+'.pickle' ,'wb')
pickle.dump(particle_orientation,filehandler500)
filehandler500.close()


#checking lenght of each elements of dictionary
##for i in range(1,460):                                       
##    print(len(particle_x[i]))
##    print(len(particle_y[i]))


##k11=particle_x[3985]
##k12=particle_y[3985]
##k13=[k11,k12]
###savetxt('C:/Users/iitk/OneDrive - IIT Kanpur/Desktop/create_video/trajectories/0nacl_3985.csv', k13, delimiter=',')
##
##k111=particle_x[3843]
##k121=particle_y[3843]
##k131=[k111,k121]
#savetxt('C:/Users/iitk/OneDrive - IIT Kanpur/Desktop/create_video/trajectories/0nacl_3843.csv', k131, delimiter=',')
#TRAJECTORY PLOT

def trajectory_plot(x_location_dict,y_location_dict):
 colors = iter(mtp.cm.rainbow(np.linspace(0, 2, 28)))
 str1=['r', 'g' ,'b' ,'c' , 'm' ,'y' ,'k','r','g' ,'b' ,'c' , 'm' ,'y' ,'k','k','r','g' ,'b' ]
 mk=iter(str1)
 p=0.1
 q=0.1
 r=0.1
 for i in range (1,11000):
  l=len(particle_x[i])
  if l>frame_tracked:
   print(i,l)
  t=np.arange(1,l+1)
 #plt.plot(t, obj[i], color=next(colors),label=str(i))
  if l>frame_tracked:  
   plt.plot((np.asarray(particle_x[i]))*0.14, np.asarray(particle_y[i])*0.14, label=str(i)) # x,y trajectory
  # plt.title('Trajectory in X and Y plane', fontsize=20)
   ax=plt.subplot(111)
   #ax.legend(loc='lower left',bbox_to_anchor=(1 ,0))
   plt.xticks(fontsize=43)
   plt.yticks(fontsize=43)
   #plt.xlabel('<X>'+ str(r'$\mu m$'),fontsize=28)
   #plt.ylabel("<Y>" + str(r'$\mu m$'),fontsize=28)
   plt.xlim([0,140])
  # plt.legend()    
 plt.show()

trajectory_plot(particle_x,particle_y)
k131=[particle_x[168],particle_y[168],particle_x[194],particle_y[194],particle_x[143],particle_y[143],particle_x[57],particle_y[57]]
savetxt('D:/mk/python_analyzed_results/Nacl_experiments/all_date_analysis/trajectory/1nacl.csv', k131, delimiter=',')



#mean square displacement of each particle

def mean_square_displacement():
 # MSD=[]
 # MSD_x=[]
 # MSD_y=[]
  all_displacement_x_for_delta_t=dictionary(no_of_frame)
  all_displacement_y_for_delta_t=dictionary(no_of_frame)
  all_displacement_r_for_delta_t=dictionary(no_of_frame)
  all_displacements=[]
  
  for j in range(1,no_of_frame):
   
   #all_particle_displacement_x_square=0
  # all_particle_displacement_y_square=0
   for k in range(0,no_of_frame-1):
     if k+j<no_of_frame:
      all_particle_displacement_square=0
      all_particle_displacement_x=0
      all_particle_displacement_y=0
      c=0
      for i in range(1,max_particle):
       l=len(particle_x[i])
       if k+j<len(particle_x[i]):
        if l>frame_tracked:
          displacement_square=((particle_x[i][k]-particle_x[i][k+j])**2 +(particle_y[i][k]-particle_y[i][k+j])**2)*0.0108
          displacement_x=(particle_x[i][k]-particle_x[i][k+j])*0.104
          displacement_y=(particle_y[i][k]-particle_y[i][k+j])*0.104
          #displacement_x_square=displacement_x**2
          # displacement_y_square=displacement_y**2
          all_particle_displacement_x=all_particle_displacement_x+ (displacement_x)
          all_particle_displacement_y=all_particle_displacement_y+(displacement_y)
          all_particle_displacement_square=all_particle_displacement_square+ math.sqrt(displacement_square)    #not square only r needed
          #all_particle_displacement_x_square=all_particle_displacement_x_square +  displacement_x_square
          # all_particle_displacement_y_square=all_particle_displacement_y_square +  displacement_y_square
          c=c+1
     # print(c,j,k)
      if c>0:
       #print(all_particle_displacement_x)
       all_displacement_x_for_delta_t[j].append( all_particle_displacement_x/c)
       all_displacement_y_for_delta_t[j].append(all_particle_displacement_y/c)
       all_displacement_r_for_delta_t[j].append(all_particle_displacement_square/c)
     #avg_particle_displacment_square=all_particle_displacment_square/(c)
    # avg_particle_displacement_x_square=all_particle_displacement_x_square /c
    # avg_particle_displacement_y_square= all_particle_displacement_y_square /c
   
   #MSD.append((avg_particle_displacment_square))
   #MSD_x.append((avg_particle_displacement_x_square))
   #MSD_y.append((avg_particle_displacement_y_square))
   
   

   #calculating for each_particle_now
  MSD_each_particle=dictionary(max_particle)
  MSD_x_each_particle=dictionary(max_particle)
  MSD_y_each_particle=dictionary(max_particle)
  for t1 in range(1,max_particle):
     l1=len(particle_x[t1])
     
     if l1>frame_tracked:
       for t2 in range(1,no_of_frame):
         dis_square_one_particle=0
         all_displacement_1_x_square=0
         all_displacement_1_y_square=0
         c1=0
         for t3 in range(0,no_of_frame-1):
           if t2+t3<len(particle_x[t1]):
             displacement_square1=((particle_x[t1][t3]-particle_x[t1][t2+t3])**2 +(particle_y[t1][t3]-particle_y[t1][t2+t3])**2)*0.0108
             displacement_1_x= (particle_x[t1][t3]-particle_x[t1][t2+t3])*0.104
             displacement_1_y= (particle_y[t1][t3]-particle_y[t1][t2+t3])*0.104
             displacement_1_x_square= displacement_1_x**2
             displacement_1_y_square=  displacement_1_y**2
             dis_square_one_particle=dis_square_one_particle+ displacement_square1
             all_displacement_1_x_square= all_displacement_1_x_square + displacement_1_x_square
             all_displacement_1_y_square= all_displacement_1_y_square + displacement_1_y_square
             c1=c1+1
         if c1>0:
          dis_square_one_particle= dis_square_one_particle/c1
          all_displacement_1_x_square=all_displacement_1_x_square/c1
          all_displacement_1_y_square=all_displacement_1_y_square/c1
          MSD_each_particle[t1].append(dis_square_one_particle)
          MSD_x_each_particle[t1].append(all_displacement_1_x_square)
          MSD_y_each_particle[t1].append(all_displacement_1_y_square)
         


  #saving the data
  print(len(MSD_each_particle[10]))
  # change this only
  #np.save('output_0nacl_msd/MSD_ensemble_average_0_per_nacl_frame_'+str(frame)+'.npy', MSD)#change the name
  
 # np.save('output_0nacl_msd_x/MSD_x_ensemble_average_0_per_nacl_frame_'+str(frame)+'.npy', MSD_x)
  #np.save('output_0nacl_msd_x/MSD_y_ensemble_average_0_per_nacl_frame_'+str(frame)+'.npy', MSD_y)
  
  filehandler1= open(saving_location+'/displacements_distribution_x_'+str(frame)+'.pickle' ,'wb')
  pickle.dump(all_displacement_x_for_delta_t,filehandler1)
  filehandler1.close()
  
  filehandler2=open(saving_location+'/displacements_distribution_y_'+str(frame)+'.pickle' ,'wb')
  pickle.dump(all_displacement_y_for_delta_t,filehandler2)
  filehandler2.close()

  filehandler5= open(saving_location+'/displacements_distribution_r_'+str(frame)+'.pickle' ,'wb')
  pickle.dump(all_displacement_r_for_delta_t,filehandler5)
  filehandler5.close()
  
  filehandler3=open(saving_location+'/MSD_x_each_particle_'+str(conc)+'_per_nacl_'+str(frame)+'.pickle', 'wb')
  pickle.dump(MSD_x_each_particle,filehandler3)
  filehandler3.close()
  
  filehandler4=open(saving_location+'/MSD_y_each_particle_'+str(conc)+'_per_nacl_'+str(frame)+'.pickle' ,'wb')
  pickle.dump(MSD_y_each_particle,filehandler4)
  filehandler4.close()
  
  filehandler = open(saving_location+'/MSD_each_particle_'+str(conc)+'_per_nacl_'+str(frame)+'.pickle','wb')   # change the name
  pickle.dump(MSD_each_particle, filehandler)
  filehandler.close()
  


  print('mk')                                                                               
  #print('MSD',len(MSD))
 # print('MSD', len(MSD_x))
  for t5 in range(1,max_particle):
    l4=len(MSD_each_particle[t5])
    if l4>frame_tracked:
       plt.plot((np.arange(1,len(MSD_each_particle[t5])+1))/10,MSD_each_particle[t5], label=str(t5), marker='o')
  #plt.plot((np.arange(1,no_of_frame)),MSD,label='average', marker='+')
  plt.yscale('log')
  plt.xscale('log')
  plt.title('Mean square displacement',fontsize=18)
  plt.xlabel(' Δt', fontsize=18)
  plt.ylabel(' MSD', fontsize=18)
  plt.legend()
  plt.show()
mean_square_displacement()


##MSD1= np.load('MSD_all-particle-1030-1820.npy')
##displacements_list=np.load('dispacemnents_list-1030-1820.npy')
##file_to_read = open("MSD_each_particle-1030-1820.pickle", "rb")
##loaded_dictionary = pickle.load(file_to_read)
#plt.plot(np.arange(0.1, 74.5,0.1), MSD1)
#plt.show()

##
##for t5 in range(1,460):
##    l4=len(loaded_dictionary[t5])
##    if l4>1:
##       plt.plot((np.arange(1,745)),loaded_dictionary[t5], label=str(t5), marker='o')
##plt.plot((np.arange(1,745)),MSD1,label='average', marker='+')
##plt.yscale('log')
##plt.xscale('log')
##plt.title('Mean square displacement',fontsize=18)
##plt.xlabel(' Δt', fontsize=18)
##plt.ylabel(' MSD', fontsize=18)
##plt.legend()
##plt.show()



def lin_fit(x_array, y_array):
     
     slope=(len(x_array)*np.sum(x_array*y_array)-np.sum(x_array)*np.sum(y_array))/(len(x_array)*np.sum(x_array*x_array)-np.sum(x_array)*np.sum(x_array))
     constant=(np.sum(y_array)*np.sum(x_array*x_array)-np.sum(x_array)*np.sum(x_array*y_array))/(len(x_array)*np.sum(x_array*x_array)-np.sum(x_array)*np.sum(x_array))
     #plt.plot(x_array, slope*time+ constant)
     #plt.show()
     return(slope, constant)
#slope,constant= lin_fit(time, array_11)




def diffusion_coefficients():
 time=np.arange(0,74.4,0.1)
 Diffusion_list=[]
 for i in range(1,460):
    l10 = len(loaded_dictionary[i])
    if l10 > 1:
      slope,constant=lin_fit(time, np.asarray(loaded_dictionary[i]))
      #print(slope)
      Diffusion_list.append(slope/4)
    i=i+1
 return Diffusion_list

##Diffusion_list= diffusion_coefficients()
###print(Diffusion_list)
##plt.hist( Diffusion_list, 100)
##plt.title('diffusion coefficient distribution')
##plt.ylabel('Frequency')
##plt.xlabel('diffusion coefficients (mm^2/s')
##plt.show()
def symetric_binning( list, width_of_bin,upper_limit,lower_limit):
  binned=[]
  lm=lower_limit
  while lm<= upper_limit:
    counter=0
    for i in (list):
        if lm-width_of_bin*0.5  <=i< lm+width_of_bin*0.5:
         counter=counter+1
    binned.append(counter)
    lm=lm+width_of_bin
  return(binned)


  



def probablity_displacement():
 
  bin_dis=symetric_binning(displacements_list,1,49,-80)
  bin_dis=np.asarray(bin_dis)
  bin_dis=bin_dis/3325680
  bins=np.arange(-80,50,1)
  np.save('displacemet_distribution-1030-1820.npy' ,bin_dis)
  
  plt.plot(bins,bin_dis)
  
  plt.title('probability distribution of the displacements')
  plt.xlabel( 'displacement (mm)')
  plt.ylabel( 'Probablity')
  plt.show()


#print((displacements_list))
#print(len(displacements_list))
#probablity_displacement()

def displacment_distribution():
 def gaussian_fitting(xdata, ydata):
     parameters, covariance = curve_fit(Gauss, xdata, ydata)
     return  parameters, covariance

 Y=np.load('displacemet_distribution-1030-1820.npy')
 bins=np.arange(-80,50,1)
 def Gauss(x, A, B):
    y = A*np.exp(-1*B*x**2)
    return y
  
 parameters, covariance= gaussian_fitting(bins,Y )

  
 fit_y = Gauss(bins, parameters[0], parameters[1])
 plt.plot(bins, Y, 'o', label='data', marker='o')
 plt.plot(bins, fit_y, '-', label='fit', marker='*')
 plt.title('probability distribution of the displacements')
 plt.xlabel('diplacemets (mm)')
 plt.ylabel('probablity')
 plt.legend()
 plt.show()
 Y=np.asarray(Y)
 plt.plot(bins, Y,marker='+')
 plt.title('probability distribution of the displacements')
 plt.xlabel('diplacemets (mm)')
 plt.ylabel('probablity')
 plt.yscale('log')
 plt.show()


#pair correlation function
def pair_correaltion_function(no_of_frame,no_of_particle):
    
  distance1=[]
  for t in range(0,no_of_frame):

   for i in range(1,749):
    for j in range(i+1,750):
     if len(particle_x[i])>0 and len(particle_x[j])>0:
       d=np.sqrt((particle_x[i][t]-particle_x[j][t])**2+(particle_y[i][t]-particle_y[j][t])**2)
       distance1.append(d)
  print(len(distance1))
  
  particle_distribution= create_bin(distance1,1,91)
  print(len(particle_distribution))
  particle_distribution=np.asarray(particle_distribution)
  avg_particle_distribution=particle_distribution/(no_of_frame*no_of_particle)
  gr=[]
  for i in range(len(avg_particle_distribution)):
      j=i*5
      f=avg_particle_distribution[i]/(3.14*(j**2-(j-5)**2))
      gr.append(f)

  print(len(gr))
  a1=np.arange(0,91,1)
  plt.plot(a1,gr)
  plt.title('Pair Correlation Function', fontsize=18)
  plt.xlabel(' radial distance in mm',fontsize=18)
  plt.ylabel('g(r)',fontsize=18)
  plt.show()
  return(gr,distance1)


#distance_distribution,distance= pair_correaltion_function(744,12)
#print(distance,distance_distribution)
  

def fourier_transformation(discrete_fn, width):
  laplace_transformed=[]
  for k in range(0,744):
     s=k*0.0067
     area=0
     for i in range (1,(len(discrete_fn)-1)):
       Int= discrete_fn[i]*2.71**(-1j*s*i*0.1)*width
       #Int= discrete_fn[i]*2.71**(-1j*s*((i+1)/10000))*width
       area=Int+area
     I= (width/2)*( discrete_fn[0]*2.718**(-1j*s*0)+ discrete_fn[len(discrete_fn)-1]*2.718**(-1j*74.4*s))+ area
     #I= (width/2)*( discrete_fn[0]*2.718**(-1j*s*(1)/10000)+ discrete_fn[len(discrete_fn)-1]*2.718**(-1j*s*(9999)/10000))+ area
     laplace_transformed.append(I)
    
  return(laplace_transformed)

#laplace_transformed= fourier_transformation( MSD1, 0.1)
#print('laplace_transformed', len(laplace_transformed))


def fast_fourier_transformation( input_array):
 N = 744
# sample spacing
 #T = 1.0 / 800.0
 #x = np.linspace(0.0, N*T, N, endpoint=False)
 #y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
 MSD1=np.asarray(input_array)
 yf = fft(MSD1 , 0.1)
 print(len(yf))
 xf = fftfreq(744, 0.1)[:N//2]
 print(len(xf))
 import matplotlib.pyplot as plt
 plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
 plt.grid()
 plt.show()
 return(yf)
#yf =fast_fourier_transformation(MSD1)
##frequency= np.fft.fftfreq(len(yf))
##print(frequency)
##print('yf', len(yf))



#print( fftfreq(50,1))
def storage_loss_calculation(fourier_transformed_array):
 storage=[]
 loss=[]
 N=744
 for i in range(1,len(fourier_transformed_array)+1):
  g=(0.0259/(3*13.2*3.14*i*1j*(fourier_transformed_array[i-1])))
  storage.append(abs(g.real))
  loss.append(abs(g.imag))
 print(len(storage))
 plt.scatter(fftfreq(744, 0.1)[:N//2],storage[0:N//2], label='storage modulus')
 #plt.scatter(np.linspace(0,0.0067,744),storage, label='storage modulus')
 #plt.scatter(np.arange(10,1000,10),loss, label=' loss modulus')
 plt.scatter(fftfreq(744, 0.1)[:N//2],loss[:N//2], label='loss modulus')
 #plt.scatter(np.linspace(0,0.0067,744),loss, label='loss modulus')
 plt.yscale('log')
 plt.xscale('log')
 plt.title(' frequency dependent storage and loss modulus')
 plt.xlabel(' omega ( sec^-1)')
 plt.ylabel( " G'( omega), G''( omega)  (ev/ mm^3)")
 plt.legend()
 plt.show()
 return storage[1:N//2]

 
#storage=storage_loss_calculation(yf)

#storage=np.asarray(storage)
##x=np.asarray(fftfreq(744,0.1)[1:744//2])
##print(x)
##print(storage)
##print(np.shape(x))
##print(np.shape(storage))
###plt.scatter(x, (storage))
###plt.xscale('log')
###plt.yscale('log')
##plt.show()
##x1=np.log10(x)
##print(x1)
##storage1=(np.log10(storage))
##plt.scatter(x1, storage1)
##plt.show()


#velocity at all time
def calc_velocity():
    
 speed=dictionary(30)
 for i in range(1,30):
    if len(particle_x[i])>0:
        for j in range(0,11):
          v= np.sqrt((particle_x[i][j]-particle_x[i][j+1])**2+(particle_y[i][j]-particle_y[i][j+1])**2)
          speed[i].append(v)
 print(len(speed[6]))
 return(speed)

#speed=calc_velocity()
#def velocity_correlation ():

