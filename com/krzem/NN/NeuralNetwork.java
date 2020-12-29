package com.krzem.NN;



import java.io.BufferedReader;
import java.io.FileReader;
import java.io.PrintWriter;
import java.lang.Math;



public class NeuralNetwork{
	public int i;
	public int[] h;
	public double[][][] wl;
	public double[][] bl;
	public double lr;



	public NeuralNetwork(int input,int[] hidden,int output,double lr){
		this.i=input;
		this.h=new int[hidden.length+1];
		for (int i=0;i<hidden.length;i++){
			this.h[i]=hidden[i];
		}
		this.h[hidden.length]=output;
		this.lr=lr;
		this.wl=new double[this.h.length][][];
		this.bl=new double[this.h.length][];
		for (int i=0;i<this.h.length;i++){
			int s=(i==0?this.i:this.h[i-1]);
			int e=this.h[i];
			this.wl[i]=new double[s][e];
			for (int j=0;j<s;j++){
				for (int k=0;k<e;k++){
					this.wl[i][j][k]=Math.random();
				}
			}
			this.bl[i]=new double[e];
			for (int j=0;j<e;j++){
				this.bl[i][j]=Math.random();
			}
		}
	}



	public double[] predict(double[] a){
		for (int i=0;i<this.h.length;i++){
			double[] na=new double[this.wl[i][0].length];
			for (int j=0;j<this.wl[i][0].length;j++){
				for (int k=0;k<this.wl[i].length;k++){
					na[j]+=this.wl[i][k][j]*a[k];
				}
				na[j]=1/(1+Math.exp(-na[j]-this.bl[i][j]));
			}
			a=na;
		}
		return a;
	}



	public void train(double[] a,double[] t){
		double[][] ol=new double[this.h.length+1][];
		double[] e=new double[t.length];
		ol[0]=a;
		for (int i=0;i<this.h.length;i++){
			ol[i+1]=new double[this.wl[i][0].length];
			for (int j=0;j<this.wl[i][0].length;j++){
				for (int k=0;k<this.wl[i].length;k++){
					ol[i+1][j]+=this.wl[i][k][j]*ol[i][k];
				}
				ol[i+1][j]=1/(1+Math.exp(-ol[i+1][j]-this.bl[i][j]));
				if (i==this.wl.length-1){
					e[j]=t[j]-ol[i+1][j];
				}
			}
		}
		for (int i=this.h.length-1;i>=0;i--){
			double[] g=new double[this.bl[i].length];
			for (int j=0;j<g.length;j++){
				g[j]=ol[i+1][j]*(1-ol[i+1][j])*e[j]*this.lr;
				this.bl[i][j]+=g[j];
			}
			for (int j=0;j<this.wl[i].length;j++){
				for (int k=0;k<this.wl[i][0].length;k++){
					this.wl[i][j][k]+=g[k]*ol[i][j];
				}
			}
			double[] ne=new double[this.wl[i].length];
			for (int j=0;j<this.wl[i][0].length;j++){
				for (int k=0;k<this.wl[i].length;k++){
					ne[k]+=e[j]*this.wl[i][k][j];
				}
			}
			e=ne;
		}
	}



	public void train_multiple(double[][][] d,double[][][] td,int itr,int s,int l_steps,boolean log){
		int p=0;
		for (int i=s+0;i<itr;i++){
			p=(int)Math.floor((double)(i)/itr*l_steps);
			String n=Double.toString((double)(p)/(l_steps/100d)).replace("\\.0$","");
			String str=String.format("Epoch %d/%d (%s%%), %f%% acc",i,itr,n,this.acc(td)*100);
			while (str.length()<100){
				str+=" ";
			}
			System.out.println(str);
			int ep=-1;
			long ost=-1;
			long st=-1;
			for (int k=0;k<d.length;k++){
				if (ep<(int)((double)k/d.length*10000)){
					ep=(int)((double)k/d.length*10000);
					double sd=(double)(System.nanoTime()-ost);
					double pd=(double)(System.nanoTime()-st);
					if (ost==-1||st==-1){
						ost=System.nanoTime();
						sd=0;
						pd=0;
					}
					for (int l=0;l<100;l++){
						System.out.print("\b");
					}
					String ds="";
					int m=(int)((double)ep/500);
					for (int l=0;l<20;l++){
						if (l<m){
							ds+="=";
						}
						else if (l==m){
							ds+=">";
						}
						else{
							ds+=".";
						}
					}
					str=String.format("%.2f%% [%s] (t=%.1fs, dt=%.1fs, ETA=%ds)",(double)ep/100,ds,sd*1e-9,pd*1e-9,(int)(sd*1e-9/((double)ep*1e-4)-sd*1e-9));
					for (int l=0;l<100-str.length();l++){
						str+=" ";
					}
					System.out.print(str);
					st=System.nanoTime();
				}
				this.train(d[k][0],d[k][1]);
			}
			for (int k=0;k<100;k++){
				System.out.print("\b");
			}
		}
	}



	public void toFile(String p){
		try{
			PrintWriter w=new PrintWriter(p,"UTF-8");
			String hs="";
			for (int i=0;i<this.h.length;i++){
				hs+=String.format("x%d",this.h[i]);
			}
			hs=hs.substring(1);
			w.println(String.format("%f:%dx%s",this.lr,this.i,hs));
			for (int i=0;i<this.wl.length;i++){
				w.print(String.format("%dx%d:",this.wl[i][0].length,this.wl[i].length));
				for (int y=0;y<this.wl[i].length;y++){
					String s=(y>0?";":"");
					for (int x=0;x<this.wl[i][0].length;x++){
						s+=String.format("%s%f",(x==0?"":","),this.wl[i][y][x]);
					}
					w.print(s);
				}
				w.print("\n");
			}
			w.println();
			for (int i=0;i<this.bl.length;i++){
				w.print(String.format("%d:",this.bl[i].length));
				String s="";
				for (int x=0;x<this.bl[i].length;x++){
					s+=String.format("%s%f",(x==0?"":","),this.bl[i][x]);
				}
				w.print(s+"\n");
			}
			w.close();
		}
		catch (Exception e){
			e.printStackTrace();
		}
	}



	public double acc(double[][][] d){
		double s=0;
		for (int k=0;k<d.length;k++){
			double[] o=this.predict(d[k][0]);
			for (int i=0;i<o.length;i++){
				s+=1-Math.abs(o[i]-d[k][1][i]);
			}
		}
		return s/(d.length*this.h[this.h.length-1]);
	}



	public static NeuralNetwork fromFile(String p){
		try{
			BufferedReader r=new BufferedReader(new FileReader(p));
			String l=null;
			int s=0;
			int wi=0;
			int bi=0;
			NeuralNetwork nn=null;
			int[] _t_l=new int[1];
			while ((l=r.readLine())!=null){
				if (s==0){
					double lr=Double.parseDouble(l.split(":")[0]);
					l=l.split(":")[1];
					int i=-1;
					int[] h=new int[l.length()-l.replace("x","").length()+1-2];
					int o=-1;
					int hi=0;
					int idx=0;
					for (String d:l.split("x")){
						if (idx==0){
							i=Integer.parseInt(d);
						}
						else if (idx==l.length()-l.replace("x","").length()){
							o=Integer.parseInt(d);
						}
						else{
							h[hi]=Integer.parseInt(d);
							hi++;
						}
						idx++;
					}
					nn=new NeuralNetwork(i,h,o,lr);
				}
				else if (s==1&&!l.equals("")){
					int mw=Integer.parseInt(l.split(":")[0].split("x")[0]);
					int mh=Integer.parseInt(l.split(":")[0].split("x")[1]);
					nn.wl[wi]=new double[mh][mw];
					l=l.split(":")[1];
					_t_l[0]=0;
					String[] _ll=l.split(";");
					for (int y=0;y<mh;y+=20){
						int sy=y+0;
						_t_l[0]++;
						double[][] a=nn.wl[wi];
						String[] ll=_ll;
						new Thread(new Runnable(){
							@Override
							public void run(){
								for (int y=sy+0;y<Math.min(sy+20,mh);y++){
									String ln=ll[y];
									for (int x=0;x<mw;x++){
										a[y][x]=Double.parseDouble(ln.split(",")[x]);
									}
								}
								_t_l[0]--;
							}
						}).start();
					}
					while (_t_l[0]>0){
						try{
							Thread.sleep((int)(1000/30));
						}
						catch (Exception e){
							e.printStackTrace();
						}
					}
					wi++;
				}
				else if (s==2){
					int ms=Integer.parseInt(l.split(":")[0].split("x")[0]);
					nn.bl[bi]=new double[ms];
					l=l.split(":")[1];
					for (int x=0;x<ms;x++){
						nn.bl[bi][x]=Double.parseDouble(l.split(",")[x]);
					}
					bi++;
				}
				if (s==0){
					s=1;
				}
				else if (l.equals("")){
					s=2;
				}
			}
			r.close();
			return nn;
		}
		catch (Exception e){
			e.printStackTrace();
		}
		return null;
	}
}