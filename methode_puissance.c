#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

/*##################################################################################################################
	Les fonctions de calculs ; normalisation de vecteur , multiplication matricielle, multiplication vectoriel,
	multiplication d'un vecteur par un scalaire, et soustraction de vecteurs.
##################################################################################################################*/

/*
 Fonction permettant de normaliser un vecteur 
 *@param taille_vecteur : la taille du vecteur à normaliser
 *@param vecteur : le vecteur à normaliser
*/
float normaliser_vecteur(float *vecteur, int taille_vecteur)
{
	float somme_des_carres=0; 
	int i;
	
	#pragma omp parallel for private(i) schedule(dynamic,1)
	for(i=0; i<taille_vecteur; i++)
	{
		somme_des_carres = somme_des_carres + vecteur[i]*vecteur[i];
	}
	
	return  sqrt(somme_des_carres);
}

/*
 Fonction permettant d'effectuer une multiplication matrice x vecteur
 *@param matrice : la matrice à utiliser dans la multiplication
 *@param vecteur : le vecteur à utiliser dans la multiplication
 *@param lignes_matrices : le nombre de lignes de la matrice
 *@param colonnes_matrice : le nombre de colonnes de la matrice
 *@param taille_vecteur : la taille du vecteur à utiliser dans la multiplication
*/
float * multiplication_matricielle(float **matrice, float *vecteur, int lignes_matrice, int colonnes_matrice, int taille_vecteur)
{
	float *resultat =(float*) malloc(sizeof(float)*taille_vecteur);
	int i,j;
	
	#pragma omp parallel for private(i) shared(resultat) schedule(dynamic,1)
	for(i=0; i<colonnes_matrice; i++)
	{
		resultat[i] = 0;
		#pragma omp parallel for private(j) schedule(dynamic,1)
		for(j=0; j<lignes_matrice; j++)
		{
			resultat[i] = matrice[i][j]*vecteur[j] + resultat[i];
		}
	}
	return resultat;
}

/*
 Fonction permettant d'effectuer une multiplication vecteur x vecteur
 *@param vecteurUn : le premier vecteur à utiliser dans la multiplication
 *@param vecteurDeux : le deuxieme vecteur à utiliser dans la multiplication
 *@param lignes_matrices : le nombre de lignes de la matrice
*/
float multiplication_vectoriel(float *vecteurUn, float *vecteurDeux, int taille_vecteur)
{
	float somme=0;
	int i;
	
	#pragma omp parallel for private(i) schedule(dynamic,1)
	for(i=0; i<taille_vecteur; i++)
	{
		somme = vecteurUn[i]*vecteurDeux[i] + somme;
	}
		
	return somme;
}

/*
 Fonction permettant d'effectuer une multiplication vecteur x scalaire
 *@param vecteur : le vecteur à utiliser dans la multiplication
 *@param variable : le scalaire à utiliser dans cette multiplication
 *@param taille : la taille du vecteur à utiliser dans cette multiplication
*/
float * multiplication_vecteur_par_variable(float *vecteur, float variable, int taille)
{
	float *resultat =(float*) malloc(sizeof(float)*taille);
	int i;
	
	#pragma omp parallel for private(i) shared(resultat) schedule(dynamic,1)
	for(i=0; i<taille; i++)
	{
		resultat[i] = vecteur[i]*variable;
	}
	
	return resultat;
}

/*
 Fonction permettant d'effectuer une soustraction entre 2 vecteurs
 *@param vecteurUn : le premier vecteur à utiliser dans la soustraction
 *@param vecteurDeux : le deuxieme vecteur à utiliser dans la soustraction
 *@param taille : la taille des deux vecteurs 
*/
float * soustration_vecteurs(float *vecteurUn, float *vecteurDeux, int taille)
{
	float *resultat =(float*) malloc(sizeof(float)*taille);
	int i;
	
	#pragma omp parallel for private(i) shared(resultat) schedule(dynamic,1)
	for(i=0; i<taille; i++)
	{
		resultat[i] = vecteurUn[i] - vecteurDeux[i];
	}
	
	return resultat;
}

/*##################################################################################################################
	Fonction principale du projet - la méthode des puissances 
##################################################################################################################*/

/*
 Fonction principale permettant d'appliquer la methode des puissances
 *@param matrice_A : la matrice saisie pour la methode des puissances
 *@param nbr_colonnes_A : le nombre de colonnes de matrice_A
 *@param nbr_lignes_A : le nombre de lignes de matrice_A
 *@param vecteur_initial : le vecteur initial à utliser dans methode des puissances
 *@param tolerance : la tolerance saisie par l'utilisateur
 *@param nbr_maxi_iteration_possibles : nbr_maxi_iteration_possibles saisie par l'utilisateur
*/
void methode_puissances(float **matrice_A, int nbr_colonnes_A, int nbr_lignes_A, float *vecteur_initial, int taille_vecteur_initial, float tolerance, int nbr_maxi_iteration_possibles)
{
	float vecteur_initial_normalise = normaliser_vecteur(vecteur_initial, taille_vecteur_initial);
	float *taille_des_residus_successifs=NULL; taille_des_residus_successifs = malloc(sizeof(float)*nbr_maxi_iteration_possibles);
	float *approximations_succ_de_valeur_propre_recherchee=NULL; approximations_succ_de_valeur_propre_recherchee = malloc(sizeof(float)*nbr_maxi_iteration_possibles);
	float res;
	float *vecteur_z = (float*) malloc(sizeof(float)*taille_vecteur_initial);
	int nbr_iteration_necessaire_a_la_convergence_de_l_algo = 0;
	int i,j=0;
	float *vecteur_q = (float*) malloc(sizeof(float)*taille_vecteur_initial);
	float *vecteur_q2 = (float*) malloc(sizeof(float)*taille_vecteur_initial);
	float lam;
	float vecteur_normalise;
	float *vecteur_propre = (float*) malloc(sizeof(float)*taille_vecteur_initial);
	float *vecteur_z2 = (float*) malloc(sizeof(float)*taille_vecteur_initial);
	float vecteur_z2_normalise;
	float *vecteur_y1 = (float*) malloc(sizeof(float)*taille_vecteur_initial);
	float c;
	float *resultat_soustraction = (float*) malloc(sizeof(float)*taille_vecteur_initial);
	
	#pragma omp parallel for private(i) shared(vecteur_q,vecteur_initial,vecteur_q2,vecteur_initial_normalise) schedule(dynamic,1)
	for(i=0; i<taille_vecteur_initial; i++)
	{
		vecteur_q[i] = vecteur_initial[i] / vecteur_initial_normalise ;
		vecteur_q2[i] = vecteur_q[i];
	}
	
	res = tolerance + 1;
	
	vecteur_z = multiplication_matricielle(matrice_A, vecteur_initial, nbr_lignes_A, nbr_colonnes_A, taille_vecteur_initial);
	
	for(i=0; i < nbr_maxi_iteration_possibles; i++)
	{
	
		if(res > tolerance)
		{
		
			vecteur_normalise = normaliser_vecteur(vecteur_z, taille_vecteur_initial);
			
			#pragma omp parallel for private(i) shared(vecteur_propre,vecteur_q,vecteur_z,vecteur_normalise) schedule(dynamic,1)
			for(i=0; i<taille_vecteur_initial; i++)
			{
				vecteur_q[i] = vecteur_z[i] / vecteur_normalise ;
				vecteur_propre[i] = vecteur_q[i];
			}
			
			vecteur_z = multiplication_matricielle(matrice_A, vecteur_q, nbr_lignes_A, nbr_colonnes_A, taille_vecteur_initial);
			lam = multiplication_vectoriel(vecteur_z, vecteur_q, taille_vecteur_initial);
			vecteur_z2 = multiplication_matricielle(matrice_A, vecteur_q2, nbr_lignes_A, nbr_colonnes_A, taille_vecteur_initial);
			vecteur_z2_normalise = normaliser_vecteur(vecteur_z2, taille_vecteur_initial);
			
			#pragma omp parallel for private(i) shared(vecteur_q2,vecteur_z2,vecteur_y1,vecteur_z2_normalise) schedule(dynamic,1)
			for(i=0; i<taille_vecteur_initial; i++)
			{
				vecteur_q2[i] = vecteur_z2[i] / vecteur_z2_normalise ;
				vecteur_y1[i] = vecteur_q2[i];
			}
			
			c = multiplication_vectoriel(vecteur_y1, vecteur_propre, taille_vecteur_initial);
		
			if(c > 0.05)
			{
				resultat_soustraction=soustration_vecteurs(vecteur_z,multiplication_vecteur_par_variable(vecteur_q,lam,taille_vecteur_initial),taille_vecteur_initial);
				res = normaliser_vecteur(resultat_soustraction, taille_vecteur_initial);
				taille_des_residus_successifs[j] = res;
				approximations_succ_de_valeur_propre_recherchee[j] = lam;
				nbr_iteration_necessaire_a_la_convergence_de_l_algo = nbr_iteration_necessaire_a_la_convergence_de_l_algo  + 1;
			}
			else
			{	
				printf("Probléme de convergence\n");
				break;
			}
			j++;
		}
	}
	
	//Affichage des résultats
	printf("\n-----------------------------------------\n");
	printf("vecteur propre: \n");
	for(i=0;i<taille_vecteur_initial; i++)
	{
		printf("%f\n",vecteur_propre[i]);
	}
	printf("\n-----------------------------------------\n");
	printf("approximations succesives de la valeur propore recherchée: \n");
	for(i=0;i<j; i++)
	{
		printf("%f\n",approximations_succ_de_valeur_propre_recherchee[i]);
	}
	printf("\n-----------------------------------------\n");
	printf("nombre d'itérations nécessaire à la convergence de l'algorithme: \n");
	printf("%d\n",nbr_iteration_necessaire_a_la_convergence_de_l_algo);
	printf("\n-----------------------------------------\n");
	printf("tailles des résidus successifs: \n");
	for(i=0;i<j; i++)
	{
		printf("%f\n",taille_des_residus_successifs[i]);
	}
	
	
	free(vecteur_q);
	free(vecteur_q2);
	free(vecteur_z);
	free(vecteur_z2);
	free(vecteur_propre);
	free(vecteur_y1);
	free(resultat_soustraction);
	free(approximations_succ_de_valeur_propre_recherchee);
	free(taille_des_residus_successifs);
}

int main(void)
{
	
	float *vecteur_initial = NULL;
	int taille_vecteur_initial;
	float **matrice_A = NULL;
	int nbr_colonnes_A, nbr_lignes_A;
	int i,j;
	
	//Saisie des tailles; du vecteur et de la matrice A avec une saisie sécurisée (taille du vecteur doit etre egale au nombre de colonne de la matrice A)
	do
	{
		printf("############ LES DIMENSIONS ###########\n");
		printf("***VECTEUR***\n");
		printf("taille du vecteur initial  :\n>>> ");
		scanf("%d",&taille_vecteur_initial);
		printf("***MATRICE***\n");
		printf("matrice A - lignes  :\n>>> ");
		scanf("%d",&nbr_lignes_A);
		printf("matrice A - colonnes  :\n>>> ");
		scanf("%d",&nbr_colonnes_A);
	}while(taille_vecteur_initial != nbr_colonnes_A);
	
	//Allocation du vecteur initial
	vecteur_initial = malloc(taille_vecteur_initial*sizeof(float));
	
	
	//Allocation de la matrice A 
	matrice_A = (float**)malloc(nbr_colonnes_A*sizeof(float*));
	#pragma omp parallel for private(i) shared(matrice_A) schedule(dynamic,1)
	for(i=0;i<nbr_colonnes_A;i++)
	{
		matrice_A[i] = (float*)malloc(nbr_lignes_A*sizeof(float));
	}
	
	//Remplissage du vecteur 
	printf("########### LES VALEURS ###########\n");
	for(i=0; i<taille_vecteur_initial; i++)
	{
		printf("vecteur initial - valeur de la case %d:\n>>> ",i);
		scanf("%f",&vecteur_initial[i]);
	}
	
	//Rempliisage de la matrice A
	for(i=0;i<nbr_lignes_A;i++)
	{
		for(j=0;j<nbr_colonnes_A;j++)
		{
			printf("matrice A - valeur de la case (%d %d):\n>>> ",i,j);
			scanf("%f",&matrice_A[i][j]);
		}
	}
	printf("#################################\n");

	//Fonction de la méthode des puissances
	methode_puissances(matrice_A,nbr_colonnes_A,nbr_lignes_A,vecteur_initial,taille_vecteur_initial, 0.00000001, 100);

	//Désallocation vecteur + matrice A
	free(vecteur_initial);
	for(int i; i < nbr_lignes_A; i++)
	{
		free(matrice_A[i]);
	}
	free(matrice_A);
	
	return 0;
}
