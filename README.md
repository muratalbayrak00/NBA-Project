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
 => Verimizde eksiklik oldugunu dusunuyoruz. Bir macta gerceklesen sayilar gercek maclar ile tutarli degil. 

 =>


---------- YAPILACAKLAR -----------
[ ] State lere match score eklenmeli. 

[ ]