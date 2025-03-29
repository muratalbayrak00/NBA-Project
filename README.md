# NBA-Project
git pull origin main => bu maindeki degisikligi develop branchine tasir.
git pull --rebase origin main => bu ise developtaki degisikligi maine merge eder. 


Eğer git pull yerine daha manuel bir işlem yapmak istersen, önce fetch yapıp sonra merge edebilirsin.
git checkout develop
git fetch origin
git merge origin/main

Merge: Eğer main dalındaki değişiklikleri develop dalına birleştirmek istiyorsan ve tarihçeyi (commit history) korumak istiyorsan, merge kullan.

Rebase: Eğer develop dalındaki değişiklikleri main dalının üzerine taşımak ve daha temiz bir tarihçe istiyorsan, rebase kullan.

----------   SORULAR   ------------



------------------ YAPILACAKLAR ----------------
[ ] period degisimlerinde pota yonu degismeli

[x] State lere match score eklenmeli. ( match skoru yeni bulunan veride var oradan bakilacak)

[ ] rakip basket attigin ceza yemeliyiz. 

[x] period bilgisi state e eklenmeli basket atan takimi ayirt etmek icin ( period bilgisi tolgalarin veride var bunu ekleriz)

[x] oyuncu id lerinde sikinti olabilir bunlari incele 

[ ] egitim yapilirken iki takim oldugu icin ayri ayri mi dusunulmeli nasil planlanmali bu surec burasi onemli. 
