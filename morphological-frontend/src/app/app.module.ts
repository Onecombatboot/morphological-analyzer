import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { HttpClientModule } from '@angular/common/http';
import { FormsModule } from '@angular/forms';

import { AppComponent } from './app.component';

@NgModule({
  declarations: [
    AppComponent
  ],
  imports: [
    BrowserModule,
    HttpClientModule, // Allows us to talk to Spring Boot
    FormsModule       // Allows us to read text from input boxes
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }