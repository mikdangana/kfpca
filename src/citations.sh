#!/bin/sh

boutfile=bibliography.txt
bibfile=bibfile.bib


get_cite() {
    c="$1"
    txfile="$2"
    if [[ "$c" =~ ^[0-9]*$ ]]; then
	c=`grep "bibitem{" "$txfile" | awk '{print $1}' | head -${c} | tail -1 | sed 's/.*bibitem{//' | sed 's/}.*//'`
    fi
    citation=$c
    echo $c
}


get_bibitem() {
    get_cite "$1" "$2"
    cite=$citation
    tmpfile=bibitem.txt
    #echo citation = $cite, bibfile = $bibfile
    n=`wc -l $bibfile`; n=${n// */}
    start=`grep -n $cite $bibfile`; start=${start//:*/}; 
    tmp=`tail -$((n-start+1)) $bibfile | grep -n "@"`;
    end=`tail -$((n-start+1)) $bibfile | grep -n "@"`; end=${end//:*/}
    #echo "head.end = $end, start = $start, n = $n, cite = $cite"
    lines=`tail -n $((n-start)) $bibfile | head -$((end-1))`
    #echo cite = $cite, start = $start, end = $end, lines = $lines
    echo "@article{$cite," > $tmpfile; echo "$lines" >> $tmpfile
    toks=`grep -n "=" $tmpfile | awk '{gsub(" ", "<space>"); print $0}'`
    declare -A kv=([volume]="<vol>",[journal]="<jnl>")
    for l in $toks; do
        l=${l//<space>/ }; k=${l//=*/}; k=${k//*:/}; k=${k// /};
	v=${l//*=/}; v=${v//[\{\},]/}; v=`echo $v | xargs -0`
	kv["$k"]=$v; #echo "k = $k, v = $v"
    done
    kv[author]=${kv[author]// and/,}; kv[author]=${kv[author]// &/,}
    for k in journal title booktitle; do kv["$k"]=${kv["$k"]// & / \\& }; done
    item="\bibitem{$cite} ${kv[author]}, \"${kv[title]}\"."
    item="$item In: \textit{${kv[journal]}}. In: \textit{${kv[booktitle]}}"
    item="$item ${kv[volume]}, (${kv[year]}), pp. ${kv[pages]}."
    item="$item ISBN: ${kv[isbn]}. DOI: ${kv[doi]}. URL: ${kv[url]}."
    # clean up
    item=${item//<[^>]*>[,.]/}; item=${item//DOI: ./}; item=${item//URL: ./}
    item=${item//pp. ./}; item=${item//ISBN: ./}; 
    item=${item//. In: \\textit\{\}/}
    echo $item >> $boutfile
}


get_citations() { 
    txfile=conference_041818.tex; if [ $# -gt 0 ]; then txfile="$1"; fi
    echo "# = $#, txfile = $txfile"
    outfile="$txfile.txt"
    seen=""; collision=""
    i=0
    txt=`cat "$txfile"`
    echo "txt = ${#txt}"
    echo "\begin{thebibliography}{9}" > $boutfile
    for c in `grep "\\cite{[^}]*}" "$txfile"`; do
      if [[ "$c" =~ .*cite.* ]]; then
	echo "c = $c"
	c=${c//*{/}; c=${c//\}*/}
	oldc=$c
	if [[ ! "$seen" =~ .*\<$oldc\>.* || "$collision" =~ .*\<$oldc-1\>.* ]]; then 
	    i=$((i+1))
	    echo oldc = $oldc, i = $i
	    if [ $oldc != "$i" ]; then 
	        txt=${txt//cite\{$i\}/cite\{$i-1\}}
		collision="$collision,<$i-1>"
	    fi
	    if [[ "$collision" =~ .*\<$oldc-1\>.* ]]; then
	        txt=${txt//cite\{$oldc-1\}/cite\{$i\}}
            else
	        txt=${txt//cite\{$oldc\}/cite\{$i\}}
	    fi
	    get_bibitem $c $txfile
	fi
	seen="$seen,<$oldc>"
      fi
    done
    #txt=${txt//\\begin\{thebibliography\}*\\end\{thebibliography\}/}
    echo "\end{thebibliography}" >> $boutfile
    echo "$txt" > $outfile
    echo "Done. Output in $boutfile & $outfile"
}

if [ $# -gt 1 ]; then 
    bibfile=$2; 
    get_citations "$1"
else
    get_citations "$*"
fi

#get_cite "1" "$*"


