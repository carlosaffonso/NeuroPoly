// Multlayer Perceptron (backpropagatin com gradiente decrescente)
// Usando as funcoes internas do Scilab
// Simulação do Procesamento de polimeros
// Autor: Carlos Affonso ; Renato Sassi ; Ricardo Ferreira
// Data: 05/10/2010

// 
// X = Vetor de entrada
// d = saida desejada (escalar)
// W = Matriz de pesos Entrada -> Camada Oculta
// M = Matriz de Pesos Camada Oculta -> Camada saida
// eta = taxa de aprendizagem
// alfa = fator de momento
clear; clc;
//=====================================================================
// Dados de entrada
//=====================================================================
loadmatfile('-ascii','Polymer_dados.txt','f');
loadmatfile('-ascii','Polymer_alvos.txt');

dados = Polymer_dados; // Vetores de entrada 
alvos = Polymer_alvos; // Saidas desejadas correspondentes

dados=dados'
alvos=alvos'
// Número de nós da camada de saída

No=1

// Dimensão dos dados de entrada

[LinD,ColD] = size(dados);    

//====================================================================

// Embaralha vetores de entrada e saidas desejadas


// Normaliza componentes para media zero e variancia unitaria
mi = mean(dados,2);  // Media das ao longo das colunas
di = stdev(dados,2);     // desvio-padrao das colunas

for j = 1:ColD
  dados(:,j) = (dados(:,j)-mi)./di;
end;

Dn = dados;

// Define tamanho dos conjuntos de treinamento/teste (hold out)
ptrn = 0.8;      // Porcentagem usada para treino
ptst = 1-ptrn;   // Porcentagem usada para teste

J = floor(ptrn*ColD);

// Vetores para treinamento e saidas desejadas correspondentes
P = Dn(:,1:J);
T1 = alvos(:,1:J);
[lP,cP] = size(P);   // Tamanho da matriz de vetores de treinamento

// Vetores para teste e saidas desejadas correspondentes
Q = Dn(:,J+1:$);
T2 = alvos(:,J+1:$);
[lQ,cQ] = size(Q);   // Tamanho da matriz de vetores de teste

// DEFINE ARQUITETURA DA REDE
//===========================
Ne = 500; // No. de epocas de treinamento
Nr = 1;   // No. de rodadas de treinamento/teste
Nh = 8;  // No. de neuronios na camada oculta

eta = 0.01;  // Passo de aprendizagem
mom = 0.75;  // Fator de momento

for r=1:Nr,   // Inicio do loop de rodadas de treinamento
    rodada=r,
    
        // Inicia matrizes de pesos
    WW = 0.1*(2*rand(Nh,lP+1)-1);  // Pesos entrada -> camada oculta
    WW_old = WW;  // Necessario para termo de momento
    
    MM = 0.1*(2*rand(No,Nh+1)-1); // Pesos camada oculta -> camada de saida
    MM_old = MM; // Necessario para termo de momento

    
  
    // ETAPA DE TREINAMENTO
    for t = 1:Ne,

        Epoca = t;

        [s,I]=gsort(rand(1,cP));    //I  é uma permutação randômica de 1:ColD
 
        P = P(:,I); T1 = T1(:,I);  // Embaralha vetores de treinamento e saidas desejadas
        
        EQ = 0;
        for tt = 1:cP,  // Inicia LOOP de epocas de treinamento

           // CAMADA OCULTA
           X = [-1; P(:,tt)]; // Constroi vetor de entrada com adicao da entrada x0=-1
           Ui = WW*X;         // Ativacao (net) dos neuronios da camada oculta
           Yi = tanh(Ui);     // Saida entre [-1,1] (função tanh)
           
           // CAMADA DE SAIDA 
           Y = [-1;Yi];       // Constroi vetor de entrada DESTA CAMADA 

           Uk = MM*Y;        // Ativacao (net) dos neuronios da camada de saida
           Ok = tanh(Uk);     // Saida entre [-1,1] (função logistica)
           
           // CALCULO DO ERRO 
           Ek = T1(:,tt)-Ok;  // erro entre a saida desejada e a saida da rede
           EQ = EQ + 0.5*sum(Ek^2); // soma do erro quadratico de todos os neuronios
     
           // CALCULO DOS GRADIENTES LOCAIS
           Dk = 0.5*(1-Ok^2); // derivada da sigmoide logistica (camada de saida)
           DDk = Ek.*Dk; // gradiente local (camada de saida)

           Di = 0.5*(1-Yi^2);         // derivada da sigmoide logistica (camada oculta)
           DDi = Di.*(MM(:,2:$)'*DDk); // gradiente local (camada oculta)
           
             // AJUSTE DOS PESOS - CAMADA DE SAIDA
           MM_aux = MM;
           MM = MM + eta*DDk*Y' + mom*(MM-MM_old);
           MM_old = MM_aux;

           // AJUSTE DOS PESOS - CAMADA OCULTA
           WW_aux = WW;
           WW = WW + eta*DDi*X' + mom*(WW-WW_old);
           WW_old = WW_aux;
           
       end;  // Fim do loop de uma epoca
            
       EQM(r,t) = EQ/cP; // MEDIA DO ERRO QUADRATICO P/ EPOCA    
           
    end; // Fim do loop de treinamento           
           
    // ETAPA DE GENERALIZACAO  %%%
    EQ2=0;
    OUT2=[];
    SAIDA=[];           
    for tt = 1:cQ,  // Inicia LOOP de epocas de treinamento
    
           // CAMADA OCULTA
           X    = [-1; Q(:,tt)]; // Constroi vetor de entrada com adicao da entrada x0=-1
           Ui   = WW*X;          // Ativacao (net) dos neuronios da camada oculta
           Yi   = tanh(Ui);      // Saida entre [-1,1] (funcao logistica)
  
           // CAMADA DE SAIDA 
           Y = [-1;Yi];       // Constroi vetor de entrada DESTA CAMADA 
           Uk = MM*Y;         // Ativacao (net) dos neuronios da camada de saida
           Ok = tanh(Uk);     // Saida entre [-1,1] (funcao logistica)
           OUT2=[OUT2 Ok];    // Armazena saida da rede
  
           Ek = T2(:,tt)-Ok;  // erro entre a saida desejada e a saida da rede
           EQ2 = EQ2 + 0.5*sum(Ek^2); // soma do erro quadratico de todos os neuronios
           
           SAIDA=[SAIDA; norm(Ek) T2(:,tt) Ok];
     
    end;  // Fim do loop de uma epoca       
    EQM2(r)=EQ2/cQ;     // MEDIA DO ERRO QUADRATICO COM REDE TREINADA      
    
   
end  // Fim do loop de rodadas de treinamento

     // CALCULA ACERTO 

EQM_media=mean(EQM,1);  // Curva de aprendizagem media (p/ Nr realizacoes)
//plot(EQM_media); // Plota curva de aprendizagem


    // SALVA PESOS E SAÍDA

savematfile('pesos.dat','WW','-ascii');    


     // RODAR A REDE COM OS PESOS SINAPTICOS

OUT3=[];

        for tt = 1:ColD,  // Inicia LOOP de epocas de treinamento

           // CAMADA OCULTA
           X = [-1; dados(:,tt)]; // Constroi vetor de entrada com adicao da entrada x0=-1
           Ui = WW*X;         // Ativacao (net) dos neuronios da camada oculta
           Yi = tanh(Ui);     // Saida entre [-1,1] (função tanh)
           
           // CAMADA DE SAIDA 
           Y = [-1;Yi];       // Constroi vetor de entrada DESTA CAMADA 

           Uk = MM*Y;        // Ativacao (net) dos neuronios da camada de saida
           Ok = tanh(Uk);     // Saida entre [-1,1] (função logistica)
           OUT3=[OUT3 Ok];    // Armazena saida da rede
           
           // PLOTAR SAIDAS
           plot(alvos)
           plot(OUT3,'r--d')
          
           
        end
         
           
           
           
           
           


