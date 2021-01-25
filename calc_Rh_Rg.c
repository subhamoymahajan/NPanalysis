/*
This program calculates radius of gyration and hydrodyanmic radius from corrected Gromacs structure files. 
    Copyright 2021 SUBHAMOY MAHAJAN
    
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.) 

The codes used in this software have been used in publications. If you find this useful, please cite:

(1) Subhamoy Mahajan and Tian Tang, "Polyethylenimine-DNA Ratio Strongly Affects Their Nanoparticle Formation: A Large-Scale
    Coarse-Grained Molecular Dynamics Study", 2019, J. Phys. Chem. B, 123 (45), 9629-9640, DOI: https://doi.org/10.1021/acs.jpcb.9b07031

*/

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include <stdbool.h>
int ndna;
int npei;
int adna;
int apei;
int Natoms;

void print_copyright(){
    printf("#########################################################\n");
    printf("Copyright 2020 SUBHAMOY MAHAJAN\n\nThis file is part of InSilico Microscopy software\n\nInSilico Microsocpy is free software: you can redistribute it and/or modify\nit under the terms of the GNU General Public License as published by\nthe Free Software Foundation, either version 3 of the License, or\n(at your option) any later version.\n\nThis program is distributed in the hope that it will be useful,\nbut WITHOUT ANY WARRANTY; without even the implied warranty of\nMERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\nGNU General Public License for more details.\n\nYou should have received a copy of the GNU General Public License\nalong with this program.  If not, see <http://www.gnu.org/licenses/>.)\n");
} 

void print_cite(){
   printf("\nIf you are using this script please cite the following:\n"); 
   printf("(1) Mahajan, S. Tang, T., J. Phys. Chem. B, 2019, 123, 45, 9629-9640 DOI: https://doi.org/10.1021/acs.jpcb.9b07031\n \n");
    printf("#########################################################\n\n");
}


float distance(float **pos, int i, int j){
    return sqrt(powf(pos[i][0]-pos[j][0],2)+powf(pos[i][1]-pos[j][1],2)+powf(pos[i][2]-pos[j][2],2));
}

char* find_var_name(char line[1200], char delim, char varname[10]){
// finds variable name from a provided line read from the parameters file.
    int i,cnt=0;
    char *a;
    varname[0]='\0';
    for(i=0;i<1200;i++){
        if((line[i]==' ')&&(cnt==0)){//ignore starting spaces
           continue;
        }
        else if(line[i]=='\n'){//line ends: stop reading
           break;
        }
        else if(line[i]=='='){//variable name exists only before =
            varname[cnt]='\0';
            break;
        }
        else if(line[i]!=' '){//add characters to variable name.
           varname[cnt]=line[i];
           cnt+=1;
        }
    }

    a=&(line[i+1]); // returns a pointer which points after '=' symbol.
    return a;
}

void read_cluster(int fileid, char headername[100], int **cluster)
{
    FILE *f;
    char filename[200], lines[5000], foo[5];
    sprintf(filename,"%s%d.dat",headername,fileid); 
    f=fopen(filename,"r");
    int i,state,cnt=0,cid,indx; //state=0 just begun, state=1
    cid=0;
    while(fgets(lines, 5000, f)) 
    {
        state=0; 
        indx=0;
        for (i=0;i<5000;i++)
        {
           if ((lines[i]=='[')||(lines[i]==']'))
           {
               state+=1;
               if (state==2)
               {
                   cluster[cid][indx]=-1; // Ending of DNA IDs
                   indx+=1;
               }
               else if (state==4)
               {
                   cluster[cid][indx]=-2;// Ending of PEI IDs
                   break;
               }
           }
           else if ((lines[i]==' ')&&(lines[i-1]!='[')) // An ID has ended
           {
               if ((state==1)||(state==3)) // For DNA ID or PEI ID
               {
                   foo[cnt]='\0'; // Finish the ID string 'foo' with a null character
                   cluster[cid][indx]=atoi(foo);//convert the foo string to integer
                   cnt=0;
                   indx+=1;
               }
           }    
           else if (isdigit(lines[i]))
           {
               foo[cnt]=lines[i]; // Add the digits in the foo string.
               cnt+=1;
           }
           
        }
        cid+=1;
    }
    cluster[cid][0]=-3;//Clusters have ended.
    return ; 
}

void print_cluster(int **cluster)
{
   int i,j;
   for (i=0;i<ndna;i++)
   {
      printf("[ ");
      if (cluster[i][0]==-3) // No more clusters
         break;
      
      for (j=0;j<npei;j++)
      {
          if (cluster[i][j]>0)
          {
               printf("%d ",cluster[i][j]);
          }
          else if (cluster[i][j]==-1) // DNA IDs have ended
          {
               printf("] [ ");
          }
          else if (cluster[i][j]==-2) //PEI IDs have ended
               break;
      }
      printf("\n");
   }
}
void read_gro(int t,char headername[100], float **pos,float box[3])
{
// These gro files must be properly clustered.

    FILE *f;
    int cnt=0,N;
    char line[70],filename[200];
    char *pend; //Character end pointer
    char *pstart; //Charcter start pointer
    sprintf(filename,"%s%d.gro",headername,t);
    f = fopen(filename,"r");
    if (f==NULL){
       printf("GRO file %s does not exist\n",filename);
       exit(-1);
    }

    while(fgets(line, 70, f)) 
    {
        if (cnt==1){
            N=atoi(line);
        }
        if (cnt>1 && cnt<Natoms+2) //11138+2
        {
            pstart=(char*)&line+20;
            pos[cnt-2][0]=strtof(pstart,&pend);
            pos[cnt-2][1]=strtof(pend,&pend);
            pos[cnt-2][2]=strtof(pend,&pend);
        }
        else if (cnt==N+2)
        {//Not returning box at the moment
            box[0]=strtof(line,&pend);
            box[1]=strtof(pend,&pend);
            box[2]=strtof(pend,NULL);
        }
        cnt+=1;
//        if (t==1020)
//            printf("cnt:%d\n",cnt);
    } 
    fclose(f);
}

void get_Rh_Rg(float **pos, int **clusters, float *Rh, float *Rg, float *mass)
{
    float Rg2,Rh_inv,com[3],Mtot;
    int i,j,k,cnt,state,*pos_ids,ccnt=0;
    pos_ids=(int*)malloc(Natoms*sizeof(int));
    float d;
    for (i=0;i<3;i++){
        com[i]=0.0;
    } 
    for (i=0;i<ndna;i++){
        state=0;
        cnt=0; 
        if (clusters[i][0]==-3)//No more clusters
             break;
        for (j=0;j<npei+ndna+3;j++)
        {
             if (clusters[i][j]==-2) // PEI IDs have ended
                 break;
             else if (clusters[i][j]==-1) // DNA Ids have ended
                 state=1; // Hereafter reading PEIs
             else if (state==0) // Reading DNA IDs
             {
                 for (k=0;k<adna;k++)
                 {
                     pos_ids[cnt]=clusters[i][j]*adna+k; // clusters[i][j]=DNA IDs
                     if (pos_ids[cnt]>Natoms)
                     {
                         print_cluster(clusters);
                         printf("Error with pos_ids: %d,%d,%d\n",clusters[i][j],adna,k);
                         exit(-1);
                     }
                     cnt++;
                 }
             }
             else if (state==1) //Reading PEI IDs
             {
                 for (k=0;k<apei;k++)
                 {
                     pos_ids[cnt]=clusters[i][j]*apei+ndna*adna+k; //clusters[i][j] = PEI ID
                     if (pos_ids[cnt]>Natoms)
                     {
                         print_cluster(clusters);
                         printf("Error with pos_ids: %d,%d,%d,%d,%d\n",clusters[i][j],ndna,adna,apei,k);
                         exit(-1);
                     }
                     cnt++;
                 }
             }
        }
      
        if (cnt>Natoms)
        {
            printf("Error with cnt= %d\n",cnt);
            exit(-1);
        }

        Rh_inv=0;
        for (j=0;j<cnt-1;j++){
            for(k=j+1;k<cnt;k++){
//                  d=sqrt(powf(pos[pos_ids[k]][0]-pos[pos_ids[j]][0],2)+powf(pos[pos_ids[k]][1]-pos[pos_ids[j]][1],2)+powf(pos[pos_ids[k]][2]-pos[pos_ids[j]][2],2));
                  d=distance(pos,pos_ids[k],pos_ids[j]);
                  Rh_inv+=1.0/d;
            }
        }
        //Center of mass
        Mtot=0;
        for (k=0;k<3;k++){        
            for (j=0;j<cnt;j++){
                 com[k]+=pos[pos_ids[j]][k]*mass[pos_ids[j]];
                 if (k==0){
                      Mtot+=mass[pos_ids[j]];
                 }
            }
            com[k]=com[k]/Mtot;
        }
        //Calculate Rg square
        for (k=0;k<3;k++){        
            for (j=0;j<cnt;j++){
                 Rg2+=(pos[pos_ids[j]][k]-com[k])*(pos[pos_ids[j]][k]-com[k])*mass[pos_ids[j]];
            }
        }
        Rg2=Rg2/Mtot;
        Rg[ccnt]=(float)sqrt(Rg2);
        Rh[ccnt]=((float)(cnt*cnt)/Rh_inv);
//        printf("%d\n",ccnt);
        ccnt+=1; 
    }
 //   printf("here %d\n",ccnt);
    Rh[ccnt]=-1; // Signifies no more Rh 
    Rg[ccnt]=-1;
    return;
}


int main (int argc, char* argv[])
{
    print_copyright();
    print_cite();
    FILE *w1, *w2, *f;
    char line[100];
    int i,t,N, **clusters;
    float **pos, *Rh, *Rg, *mass, box[3];
    char outRh[100],outRg[100],inname[100],clustname[100],constants[100],varname[50],massname[50];
    char *a;
    int times, p_per_d;
//Arguments
/*
inname : reads the input filename 
outfile : reads the output file name
clustname : reads the name of cluster files
times : reads the number of time steps
constants: reads the name of constants file
*/
    printf("argc = %d\n",argc);
    for (i=2;i<argc;i+=2){
       if (strcmp(argv[i-1],"-i")==0){ 
           strcpy(inname,argv[i]);
           printf("input gro file header name is %s\n",inname);
       }  
       if (strcmp(argv[i-1],"-m")==0){ 
           strcpy(massname,argv[i]);
           printf("file containing mass information is %s\n",massname);
       }  
       if (strcmp(argv[i-1],"-oRh")==0){ 
           strcpy(outRh,argv[i]);
           printf("Rh output file name is %s\n",outRh);
       }  
       if (strcmp(argv[i-1],"-oRg")==0){ 
           strcpy(outRg,argv[i]);
           printf("Rg output file name is %s\n",outRg);
       }  
       if (strcmp(argv[i-1],"-clust")==0){ 
           strcpy(clustname,argv[i]);
           printf("cluster file header name is %s\n",clustname);
       }  
       if (strcmp(argv[i-1],"-cons")==0){ 
           strcpy(constants,argv[i]);
           printf("constants file name is %s\n",constants);
       }  
       if (strcmp(argv[i-1],"-times")==0){ 
           times=atoi(argv[i]);
           printf("Total number of time steps is %d\n",times);
       }  
    }
/////Read constants
    f=fopen(constants,"r");
    if (f==NULL){
       printf("Constants file %s does not exist\n",constants);
       exit(-1);
    }
    while ((fgets(line,100,f)) != NULL ){
       a=find_var_name(line,'=',varname);
       if (varname[0]=='\0'){//empty line gives NULL variable names 
           continue;
       }
       if(strcmp(varname,"ndna")==0){
           ndna=atoi(a);
           printf("ndna = %d\n",ndna);
       }
       if(strcmp(varname,"p_per_d")==0){
           p_per_d=atoi(a);
       }
       if(strcmp(varname,"adna")==0){
           adna=atoi(a);
           printf("adna = %d\n",adna);
       }
       if(strcmp(varname,"apei")==0){
           apei=atoi(a);
           printf("ndna = %d\n",apei);
       }
    }
    fclose(f);
    npei=p_per_d*ndna;
    printf("npei = %d\n",npei);
    Natoms=ndna*adna+npei*apei;
    printf("Natoms = %d\n",Natoms);
    
///Initilaize pos
    pos=(float**)malloc(Natoms*sizeof(float*));
    for (i=0;i<Natoms;i++){
        pos[i]=(float*)malloc(3*sizeof(float));
    }
//Initialize cluster
    clusters=(int**)malloc(ndna*sizeof(int*));
    for (i=0;i<ndna;i++){
        clusters[i]=(int*)malloc((npei+ndna+3)*sizeof(int));       
    }
//Initialize Rh
    Rh=(float*)malloc(ndna*sizeof(float));
//Initialize Rg
    Rg=(float*)malloc(ndna*sizeof(float));
//Initialize mass
    mass=(float*)malloc(Natoms*sizeof(float));
// Read mass
    f=fopen(massname,"r");
    i=0;
    while ((fgets(line,100,f)) != NULL ){
        if (i>=Natoms){
             break;
        }
        mass[i]=atof(line);
        i++; 
    }
    fclose(f);
    w1=fopen(outRh,"w");
    w2=fopen(outRg,"w");

    for (t=0;t<times;t++)
    {
        read_gro(t,inname,pos,box);
        read_cluster(t,clustname,clusters);
        get_Rh_Rg(pos,clusters,Rh,Rg,mass);
        fprintf(w1,"%d ",t);
        fprintf(w2,"%d ",t);
        for (i=0;i<ndna;i++)
        {
             if (Rh[i]<0){
                 break; // Error
             }
             fprintf(w1,"%.4f ",Rh[i]);
             fprintf(w2,"%.4f ",Rg[i]);
        }
        fprintf(w1,"\n");
        fprintf(w2,"\n");
 
        printf("%d Done\n",t);
    }
    fclose(w1);
    fclose(w2);
    return 0;
}
        







