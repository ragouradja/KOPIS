#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include <unistd.h>


#define MAXSIZE 4096

int start;
int end;

int ind=0;

void parse_pdb(char *); /* Protype de la fonction parse_pdb */
double dist(float,float,float,float,float,float);
void help(void);

#define MONOMER '_'
int main (int argc, char *argv[])
{


	char NAME_PDB_FILE[1024];
	char *p_name_pdb_file;
	char *p_code_pdb;
	char code_pdb[5];
	int x,y,z;

	int i;

    /* minimum required number of parameters */
#define MIN_REQUIRED 1

    if (argc < MIN_REQUIRED) 
    {
        help();
        exit (-1);
    }
    int iarg;
    int num_correct_arg=0;
    int indarg=0;
    /* iterate over all arguments */
    for (i = 1; i < (argc); i++) 
    {
        /* do something with it */
        indarg=i;
        strcpy (NAME_PDB_FILE,argv[indarg]);
        //printf("# pdb file = %s\n", NAME_PDB_FILE);
        continue;
    };

    /*Recuperation des arguments */
    /*	    strcpy (NAME_PDB_FILE,argv[1]);      //argument 1: PDB file
    */


	/* Affichage des arguments*/
	strncpy(code_pdb,NAME_PDB_FILE,5);
	code_pdb[5]='\0';	
	p_name_pdb_file=NAME_PDB_FILE;
	p_code_pdb=code_pdb;
    //printf("PDB:%s\n", p_name_pdb_file);
	/* Parsepdb */
	parse_pdb(p_name_pdb_file);

}
/*
   1--------10-------20--------30--------40--------50--------60
   +--------+--------+---------+---------+---------+---------+
   ATOM      2  CA  ALA     1      11.846  49.175  18.125  1.00 23.06       1SG   3
   */

/****************************************************************************/
/*                                                                          */
/*  FONCTION OUVERTURE ET TRAITEMENT DU PDB ET CALCUL MATRICE DE CONTACT   */
/*                                                                          */
/****************************************************************************/
void parse_pdb (char *p_name_pdb_file)
{
    /* Variables lecture fichier */
	char line[85];
	char line_out[80];
	FILE *pdb_file;
	char chain='x';
	char at, ch, aa3[4], path[128], str[128], buf[128], atname[4],aaname[3];
	double x=0;
	double y=0;
	double z=0;
	int numat;
	int c=1;/* variable contenant le nombre de chaine */

    double* tab_x;
    double* tab_y;
    double* tab_z;

    tab_x=(double *) malloc(4096 * sizeof(double));
    tab_y=(double *) malloc(4096 * sizeof(double));
    tab_z=(double *) malloc(4096 * sizeof(double));

    // Pas **tab_aa type char**, 
    char **tab_aa;
    tab_aa = malloc(4096 * sizeof(char*)); // ici c'est bien sizeof(char *) 

    int taille=3;
 	/* Variable calcul des distances */
	int i;
	int j;

    if (tab_aa == NULL)  
    {  
        printf("Impossible d'allouer la mémoire pour tab_aa !!!\n" );  
    }  

     for (i = 0; i < 4096; i++)  // là c'est i++ en dernier, pas en deuxième
    {  
        // *(code + i) ou code[i]
        tab_aa[i] = malloc((taille + 1) * sizeof(char));  

        if (tab_aa[i] == NULL)
        {  
            printf("Cannot aloaate memory for tab_aa[%d] !!!\n", i);  
        }  
        tab_aa[i][taille + 1] = '\0'; 
    }

    //int tab_x[1024],tab_y[1024],tab_z[1024];
	char tab_chain[4096];	

	/* Variable calcul des distances */
	int dx2,dy2,dz2;
	double dt,dt2;
	double tmp;

	//int tab_contact[1024][1024];

    //ALLOCATION ET INITIALISATION DE tab_pcontact
    int indexe;

	//printf("%s %s %s LIMITSIZE:%d\n",p_code_pdb,p_name_pdb_file, p_name_dssp_file, LIMITSIZESS2);


	pdb_file=fopen(p_name_pdb_file, "r"); /*Ouvrir en lecture*/
	if (pdb_file == NULL) 
	{
		printf ("Error in opening file: %s\n", p_name_pdb_file);
		exit(1);
	}

	while (!feof(pdb_file))
	{       
		fgets(line,85,pdb_file);
		if (strncmp("ATOM", line, 4)) continue; /* skip non ATOM */

		/* Execute */
		at = line[13];
		/* Chain extraction */

		sscanf(&line[6] ,"%d", &numat );
		sscanf(&line[13],"%3s", atname);
		sscanf(&line[14],"%3s", aaname);
		//printf("%s\n",atname);
        if (strncmp("CA",atname,2)) continue; /* skip non CA */
       
   		tab_aa[i] =aaname;  

		/* Si c'est la première fois que l'on rencontre le champs chaine */
		if (chain == 'x')
		{
			if (line[21]==' ')
			{
				ch = MONOMER;
			}
			else
			{
				ch = line[21];
				if (chain==MONOMER) chain=ch; /* if chain=='_', keep first chain */
			}
		}

		//printf("%i %s", ind,line);
		/* Extraction coordonnes Atom Calpha */
		line[54]=' '; sscanf(&line[46], "%lf", &z);
		line[46]=' '; sscanf(&line[38], "%lf", &y);
		line[38]=' '; sscanf(&line[30], "%lf", &x);

	        //printf("INDEX PDB: %d \n",ind);
		//printf("%d ATOM %d %-2s %3s \n",ind,numat,atname,aaname);
		//printf("%d %s %s %lf %lf %lf \n",ind,aaname,atname,x,y,z);
		strcpy(tab_aa[ind],aaname);
	        tab_chain[ind]=ch;
		tab_x[ind]=x;
		tab_y[ind]=y;
		tab_z[ind]=z;
		//printf("%d %s %s %lf %lf %lf \n",ind,aaname,atname,tab_x[ind],tab_y[ind],tab_z[ind]);
		ind++; /* Ajouter 1 compteur ind */
	}
	ind--; // elimination de la derniere incremetation car elle est vide
	fclose(pdb_file); /*Fermer le fichier */

	for (i=0 ; i <= ind ; i++)
	{
		for (j=0 ; j <= ind ; j++)
		{
            //printf("INDEX %i %i \n",i,j);
			/* Calcul de la distance inter atome*/
			//dx2=pow((tab_x[i]-tab_x[j]),2);
			//dy2=pow((tab_y[i]-tab_y[j]),2);
			//dz2=pow((tab_z[i]-tab_z[j]),2);
			//dt2=dx2+dy2+dz2;
			//dt = sqrt (dt2);

			/*Ecriture dans les fichiers de sortie */	
			//fprintf (pfile_dist_contact_mat,"%5.2f ",dt);
			//fprintf (pfile_dist_contact_mtx,"%3d %3d %5.2f\n",i+1,j+1,dt);
			//printf ("  %-d %s CA %lf %lf %lf\n",i+1,tab_aa[i],tab_x[i],tab_y[i],tab_z[i]);
			//printf ("  %-d %s CA %lf %lf %lf\n",j+1,tab_aa[j],tab_x[j],tab_y[j],tab_z[j]);

			dt=dist(tab_x[i],tab_y[i],tab_z[i],tab_x[j],tab_y[j],tab_z[j]);

			/*Calcul des proba selon un modele logistique*/
			
			//printf ("%d-%d %s %s CA CA %lf\n",i+1,j+1,tab_aa[i],tab_aa[j],dt);
			printf ("%4.2f ",dt);
		}
    	printf ("\n");
	}
}


void help()
{
    printf("Distances : a tool for compute distance between atom in a pdb file\n");
    printf("Authors: Jean-Christophe Gelly\n");
    printf("Usage: distances [pdb file] \n");
}

/* Calcul distance entre deux points i et j */
double dist (float xi,float yi,float zi,float xj,float yj,float zj) 
{
	return(sqrt( (xi-xj)*(xi-xj) + (yi-yj)*(yi-yj) + (zi-zj)*(zi-zj) ));
}
