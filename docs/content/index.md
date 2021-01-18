Welcome to the documentation of the [HH inference tools](https://gitlab.cern.ch/hh/tools/inference).

Read the [Introduction](introduction.md) for the setup instructions of this repository.
More detailed information on the different scans & fits be found in the following sections:


<style>
.container {
  position: relative;
}

.plotSlides {
  display: none;
}

.plotSlides img:hover{
  -moz-box-shadow: 1px 1px 5px #999;
  -webkit-box-shadow: 1px 1px 5px #999;
  box-shadow: 1px 1px 5px #999;
}

.cursor {
  cursor: pointer;
}

.prev,
.next {
  cursor: pointer;
  position: absolute;
  top: 40%;
  width: auto;
  padding: 16px;
  margin-top: -50px;
  color: white;
  font-weight: bold;
  font-size: 20px;
  border-radius: 0 3px 3px 0;
  user-select: none;
  -webkit-user-select: none;
}

.next {
  right: 0;
  border-radius: 3px 0 0 3px;
}

.prev:hover,
.next:hover {
  background-color: rgba(0, 0, 0, 0.8);
}

.numbertext {
  color: #f2f2f2;
  font-size: 12px;
  padding: 8px 12px;
  position: absolute;
  top: 0;
}

.caption-container {
  text-align: center;
  background-color: white;
  padding: 2px 16px;
  color: black;
}

.row:after {
  content: "";
  display: table;
  clear: both;
}

.column {
  float: left;
  width: 16.66%;
}

.demo {
  opacity: 0.6;
}

.active,
.demo:hover {
  opacity: 1;
}
</style>

<div class="container">
  <div class="plotSlides">
    <div class="numbertext">1 / 5</div>
    <a href="tasks/limits.html">
        <img src="images/limits__r__kl_n51_-25.0_25.0__fb_bbwwllvv_log.png" style="width:100%">
    </a>
  </div>

  <div class="plotSlides">
    <div class="numbertext">2 / 5</div>
    <a href="tasks/likelihood.html">
        <img src="images/nll2d__kl_n61_-30.0_30.0__kt_n41_-10.0_10.0__log.png" style="width:100%">
    </a>
  </div>

  <div class="plotSlides">
    <div class="numbertext">3 / 5</div>
    <a href="tasks/exclusion1d.html">
        <img src="images/bestfitexclusion__r_gghh__kl_n51_-25.0_25.0.png" style="width:100%">
    </a>
  </div>

  <div class="plotSlides">
    <div class="numbertext">4 / 5</div>
    <a href="tasks/pullsandimpacts.html">
        <img src="images/pulls_impacts__kl.png" style="width:100%">
    </a>
  </div>

  <div class="plotSlides">
    <div class="numbertext">5 / 5</div>
    <a href="tasks/significances.html">
        <img src="images/significances__r__kl_n17_-2.0_6.0.png" style="width:100%">
    </a>
  </div>
    
  <a class="prev" onclick="plusSlides(-1)">❮</a>
  <a class="next" onclick="plusSlides(1)">❯</a>

  <div class="caption-container">
    <p id="caption"></p>
  </div>

  <div class="row">
    <div class="column">
      <img class="demo cursor" src="images/limits__r__kl_n51_-25.0_25.0__fb_bbwwllvv_log.png" style="width:100%" onclick="currentSlide(1)" alt="Upper limits">
    </div>
    <div class="column">
      <img class="demo cursor" src="images/nll2d__kl_n61_-30.0_30.0__kt_n41_-10.0_10.0__log.png" style="width:100%" onclick="currentSlide(2)" alt="1D and 2D Likelihood scans">
    </div>
    <div class="column">
      <img class="demo cursor" src="images/bestfitexclusion__r_gghh__kl_n51_-25.0_25.0.png" style="width:100%" onclick="currentSlide(3)" alt="Combined plot: fit and exclusion">
    </div>
    <div class="column">
      <img class="demo cursor" src="images/pulls_impacts__kl.png" style="width:100%" onclick="currentSlide(4)" alt="Pulls and Impacts">
    </div>
    <div class="column">
      <img class="demo cursor" src="images/significances__r__kl_n17_-2.0_6.0.png" style="width:100%" onclick="currentSlide(5)" alt="Significances">
    </div>
    
  </div>
</div>

<script>
var slideIndex = 1;
showSlides(slideIndex);

function plusSlides(n) {
  showSlides(slideIndex += n);
}

function currentSlide(n) {
  showSlides(slideIndex = n);
}

function showSlides(n) {
  var i;
  var slides = document.getElementsByClassName("plotSlides");
  var dots = document.getElementsByClassName("demo");
  var captionText = document.getElementById("caption");
  if (n > slides.length) {slideIndex = 1}
  if (n < 1) {slideIndex = slides.length}
  for (i = 0; i < slides.length; i++) {
      slides[i].style.display = "none";
  }
  for (i = 0; i < dots.length; i++) {
      dots[i].className = dots[i].className.replace(" active", "");
  }
  slides[slideIndex-1].style.display = "block";
  dots[slideIndex-1].className += " active";
  captionText.innerHTML = dots[slideIndex-1].alt;
}
</script>


An experimental interactive datacard viewer exists too (thanks to [Benjamin Fischer](https://git.rwth-aachen.de/3pia/cms_analyses/common/-/blob/master/view_datacard.html)).
